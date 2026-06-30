import asyncio
import json
import os
import secrets
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import anthropic as _anthropic
import bcrypt
import yaml
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from yaml.loader import SafeLoader
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from analyzer import analyze_tender, distill_knowledge, SYSTEM_PROMPT
from engine import load_config, save_config, parse_message, parse_excel, compare, export_to_excel, find_suspicious_lines
from scraper import Tender, _extract_id_from_url, _extract_pdf_text_from_bytes, fetch_tender_detail, fetch_tender_list
from state import filter_new, load_seen, save_seen

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

_SIGNER = URLSafeTimedSerializer(os.environ.get("SESSION_SECRET", "et-tools-2024-change-me"))
SETTINGS_FILE = BASE_DIR / "data" / "settings.json"
KNOWLEDGE_FILE = BASE_DIR / "data" / "shared" / "knowledge.json"
_excel_cache: dict[str, bytes] = {}

app = FastAPI(title="Electra Target Tools")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.on_event("startup")
async def _init_dirs():
    for d in ["data/shared", "data/users"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)


# ── Session helpers ───────────────────────────────────────────────────────────

def _sign(username: str) -> str:
    return _SIGNER.dumps(username)

def _unsign(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    try:
        return _SIGNER.loads(token, max_age=30 * 86400)
    except (BadSignature, SignatureExpired, Exception):
        return None

def auth(et_session: Optional[str] = Cookie(default=None)) -> str:
    u = _unsign(et_session)
    if not u:
        raise HTTPException(401, "לא מחובר")
    return u

def _load_users() -> dict:
    with open(BASE_DIR / "users.yaml", encoding="utf-8") as f:
        return yaml.load(f, SafeLoader)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _udir(u: str) -> Path:
    p = BASE_DIR / "data" / "users" / u
    p.mkdir(parents=True, exist_ok=True)
    return p

def _rj(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default

def _wj(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_settings() -> dict:
    d = {
        "scraper": {
            "base_url": "https://mr.gov.il/ilgstorefront/he/search/?q=:relevance&inContract=false",
            "page_load_timeout_ms": 30000, "max_tenders_per_run": 50, "days_back": 7,
        },
        "budget": {"min_annual_ils": 500000, "max_annual_ils": 50000000},
        "industries": ["ניקיון", "אחזקה", "שמירה ואבטחה", "כוח אדם", "שירותי עזר", "קייטרינג"],
        "labor_costs": {"simple_monthly_ils": 6500, "simple_hourly_ils": 35, "social_expense_multiplier": 1.25},
    }
    if not SETTINGS_FILE.exists():
        _wj(SETTINGS_FILE, d)
        return d
    loaded = _rj(SETTINGS_FILE, d)
    for k, v in d.items():
        loaded.setdefault(k, v)
    return loaded

def _load_knowledge() -> list:
    return _rj(KNOWLEDGE_FILE, [])

def _save_knowledge(g: list):
    KNOWLEDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _wj(KNOWLEDGE_FILE, g)

def _append_history(result: dict, u: str):
    h = _rj(_udir(u) / "history.json", [])
    if result["tender_id"] not in {r["tender_id"] for r in h}:
        h.append(result)
        _wj(_udir(u) / "history.json", h)

def _client() -> _anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise HTTPException(500, "ANTHROPIC_API_KEY לא מוגדר")
    return _anthropic.Anthropic(api_key=key)


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((BASE_DIR / "static" / "index.html").read_text(encoding="utf-8"))

@app.post("/api/login")
async def login(resp: Response, username: str = Form(...), password: str = Form(...)):
    data = _load_users()
    user = data["credentials"]["usernames"].get(username)
    if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
        raise HTTPException(401, "שם משתמש או סיסמה שגויים")
    resp.set_cookie("et_session", _sign(username), max_age=30 * 86400, httponly=True, samesite="lax")
    return {"username": username, "name": user.get("name", username),
            "role": user.get("role", "user"), "apps": user.get("apps", ["tender_scanner", "shift_comparison"])}

@app.post("/api/logout")
async def logout(resp: Response):
    resp.delete_cookie("et_session")
    return {"ok": True}

@app.get("/api/me")
async def me(u: str = Depends(auth)):
    users = _load_users()["credentials"]["usernames"]
    info = users.get(u, {})
    return {"username": u, "name": info.get("name", u),
            "role": info.get("role", "user"), "apps": info.get("apps", ["tender_scanner", "shift_comparison"])}


# ── Settings & knowledge ──────────────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings(_: str = Depends(auth)):
    return _load_settings()

@app.post("/api/settings")
async def post_settings(body: dict, _: str = Depends(auth)):
    _wj(SETTINGS_FILE, body)
    return {"ok": True}

@app.get("/api/knowledge")
async def get_knowledge(_: str = Depends(auth)):
    return _load_knowledge()

@app.post("/api/knowledge")
async def post_knowledge(body: dict, _: str = Depends(auth)):
    _save_knowledge(body.get("guidelines", []))
    return {"ok": True}

@app.delete("/api/knowledge/{idx}")
async def del_knowledge(idx: int, _: str = Depends(auth)):
    k = _load_knowledge()
    if 0 <= idx < len(k):
        k.pop(idx)
        _save_knowledge(k)
    return k


# ── Tender data ───────────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history(u: str = Depends(auth)):
    return _rj(_udir(u) / "history.json", [])

@app.patch("/api/history/{tid}")
async def patch_history(tid: str, body: dict, u: str = Depends(auth)):
    path = _udir(u) / "history.json"
    h = _rj(path, [])
    for entry in h:
        if entry.get("tender_id") == tid:
            entry.update(body)
            break
    _wj(path, h)
    return {"ok": True}

@app.delete("/api/history/{tid}")
async def delete_history(tid: str, u: str = Depends(auth)):
    path = _udir(u) / "history.json"
    h = _rj(path, [])
    h = [r for r in h if r.get("tender_id") != tid]
    _wj(path, h)
    # also remove from seen so it can be re-scanned
    seen_path = _udir(u) / "seen.json"
    seen = load_seen(seen_path)
    seen.discard(tid)
    save_seen(seen, seen_path)
    return {"ok": True}

@app.get("/api/last-scan")
async def get_last_scan(u: str = Depends(auth)):
    return _rj(_udir(u) / "last_scan.json", [])

@app.get("/api/favorites")
async def get_favorites(u: str = Depends(auth)):
    return _rj(_udir(u) / "favorites.json", [])

@app.post("/api/favorites/{tid}")
async def toggle_fav(tid: str, u: str = Depends(auth)):
    favs: list = _rj(_udir(u) / "favorites.json", [])
    if tid in favs:
        favs.remove(tid)
    else:
        favs.append(tid)
    _wj(_udir(u) / "favorites.json", favs)
    return favs


# ── Scan (SSE) ────────────────────────────────────────────────────────────────

@app.get("/api/scan-test")
async def scan_test(u: str = Depends(auth)):
    import traceback
    try:
        settings = _load_settings()
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(locale="he-IL", extra_http_headers={"Accept-Language": "he-IL,he;q=0.9"})
            page = await context.new_page()
            await page.goto(settings["scraper"]["base_url"], timeout=30000, wait_until="domcontentloaded")
            await page.wait_for_timeout(3000)
            final_url = page.url
            items = await page.query_selector_all("div.result-container")
            # check show-more button with various selectors
            show_more = None
            show_more_sel = None
            for sel in ["button.show-more-button","button:has-text('הצג עוד')","a:has-text('הצג עוד')",".show-more","[class*='show-more']"]:
                el = await page.query_selector(sel)
                if el:
                    show_more = await el.inner_text()
                    show_more_sel = sel
                    break
            # get first 3 titles
            titles = []
            for item in items[:3]:
                el = await item.query_selector("h2")
                if el: titles.append(await el.inner_text())
            await browser.close()
            return {"final_url": final_url, "items_found": len(items), "show_more_btn": show_more, "show_more_sel": show_more_sel, "sample_titles": titles}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()[-1000:]}

@app.get("/api/scan")
async def scan(u: str = Depends(auth), skip_seen: bool = Query(True)):
    settings = _load_settings()

    async def gen():
        import traceback
        try:
            client = _client()
            loop = asyncio.get_running_loop()
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','msg':f'שגיאת אתחול: {traceback.format_exc()[-400:]}'})}\n\n"
            return

        yield f"data: {json.dumps({'type':'status','msg':'מתחבר ל-mr.gov.il...'})}\n\n"
        try:
            import traceback
            tender_list = await asyncio.wait_for(fetch_tender_list(settings), timeout=120)
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type':'error','msg':'timeout — הסריקה לקחה יותר מ-120 שניות'})}\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','msg': traceback.format_exc()[-500:]})}\n\n"
            return

        seen_path = _udir(u) / "seen.json"
        seen = load_seen(seen_path) if skip_seen else set()
        new = filter_new(tender_list, seen) if skip_seen else tender_list

        yield f"data: {json.dumps({'type':'status','msg':f'נמצאו {len(tender_list)} מכרזים, {len(new)} חדשים לניתוח'})}\n\n"
        yield f"data: {json.dumps({'type':'count','total':len(tender_list),'new':len(new)})}\n\n"

        if not new:
            yield f"data: {json.dumps({'type':'complete','count':0,'total':len(tender_list)})}\n\n"
            return

        results = []
        for i, meta in enumerate(new):
            yield f"data: {json.dumps({'type':'progress','i':i+1,'total':len(new),'title':meta['title'][:80]})}\n\n"
            yield ": keepalive\n\n"

            try:
                tender = await asyncio.wait_for(fetch_tender_detail(meta, settings), timeout=90)
                tender.raw_metadata.update({"publish_date": meta.get("publish_date",""), "update_date": meta.get("update_date","")})

                if not tender.pdf_text:
                    result = {
                        "tender_id": tender.tender_id, "title": tender.title, "url": tender.url,
                        "publisher": tender.publisher, "deadline": tender.deadline,
                        "publish_date": meta.get("publish_date",""), "update_date": meta.get("update_date",""),
                        "has_pdf": False, "analysis": "NO_PDF",
                    }
                else:
                    knowledge = _load_knowledge()
                    result = await loop.run_in_executor(
                        None, lambda t=tender: analyze_tender(t, settings, client, knowledge=knowledge)
                    )
            except asyncio.TimeoutError:
                result = {
                    "tender_id": meta["tender_id"], "title": meta["title"], "url": meta["url"],
                    "publisher": "", "deadline": "",
                    "publish_date": meta.get("publish_date",""), "update_date": meta.get("update_date",""),
                    "has_pdf": False, "analysis": "שגיאה: timeout",
                }
            except Exception as e:
                result = {
                    "tender_id": meta["tender_id"], "title": meta["title"], "url": meta["url"],
                    "publisher": "", "deadline": "",
                    "publish_date": meta.get("publish_date",""), "update_date": meta.get("update_date",""),
                    "has_pdf": False, "analysis": f"שגיאה: {e}",
                }

            results.append(result)
            _append_history(result, u)
            seen.add(meta["tender_id"])
            if skip_seen:
                save_seen(seen, seen_path)
            _wj(_udir(u) / "last_scan.json", results)

            yield f"data: {json.dumps({'type':'result','data':result,'pct':(i+1)/len(new)})}\n\n"

        yield f"data: {json.dumps({'type':'complete','count':len(results),'total':len(tender_list)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Analyze single URL ────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_url(body: dict, u: str = Depends(auth)):
    url = body.get("url", "").strip()
    if not url:
        raise HTTPException(400, "URL נדרש")
    settings = _load_settings()
    tid = _extract_id_from_url(url)
    meta = {"tender_id": tid, "title": url, "url": url, "publish_date": "", "update_date": ""}
    try:
        tender = await asyncio.wait_for(fetch_tender_detail(meta, settings), timeout=90)
        if not tender.title or tender.title.startswith("http"):
            tender.title = f"מכרז {tid}"
        tender.raw_metadata.update({"publish_date": "", "update_date": ""})
        if not tender.pdf_text:
            result = {
                "tender_id": tid, "title": tender.title, "url": url,
                "publisher": tender.publisher, "deadline": tender.deadline,
                "publish_date": "", "update_date": "", "has_pdf": False, "analysis": "NO_PDF",
            }
        else:
            client = _client()
            loop = asyncio.get_running_loop()
            knowledge = _load_knowledge()
            result = await loop.run_in_executor(
                None, lambda t=tender: analyze_tender(t, settings, client, knowledge=knowledge)
            )
        _append_history(result, u)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(408, "timeout")
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Analyze uploaded PDF ──────────────────────────────────────────────────────

@app.post("/api/analyze-pdf")
async def analyze_pdf_upload(pdf: UploadFile = File(...), u: str = Depends(auth)):
    body = await pdf.read()
    if not body:
        raise HTTPException(400, "קובץ ריק")
    pdf_text = _extract_pdf_text_from_bytes(body)
    if not pdf_text.strip():
        raise HTTPException(422, "לא ניתן לחלץ טקסט מהקובץ")
    settings = _load_settings()
    client = _client()
    knowledge = _load_knowledge()
    filename = pdf.filename or "מכרז"
    title = filename.replace(".pdf", "").replace("_", " ")
    tender = Tender(
        tender_id=f"pdf_{int(time.time())}",
        title=title,
        url="",
        publisher="",
        deadline="",
        raw_metadata={"publish_date": "", "update_date": ""},
        pdf_text=pdf_text,
    )
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: analyze_tender(tender, settings, client, knowledge=knowledge)
    )
    _append_history(result, u)
    return result


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(body: dict, _: str = Depends(auth)):
    tender = body.get("tender", {})
    history = body.get("history", [])
    client = _client()
    system = f"""אתה יועץ עסקי שניתח מכרז עבור Electra Target.
כותרת: {tender.get('title','')}
מפרסם: {tender.get('publisher','לא ידוע')}
מועד הגשה: {tender.get('deadline','לא צוין')}
ניתוח: {tender.get('analysis','')}
ענה בעברית בלבד. היה ממוקד ותמציתי."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    loop = asyncio.get_running_loop()
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-sonnet-4-6", max_tokens=1024, system=system, messages=msgs
            )
        )
        return {"reply": resp.content[0].text}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Re-analyze with clarification answers ─────────────────────────────────────

@app.post("/api/tender/reanalyze-with-answers")
async def reanalyze_with_answers(
    answers: UploadFile = File(...),
    tender_data: str = Form(...),
    u: str = Depends(auth)
):
    import json as _json
    td = _json.loads(tender_data)
    answers_text = (await answers.read()).decode("utf-8", errors="ignore")
    settings = _load_settings()
    client = _client()
    knowledge = _load_knowledge()
    tender = Tender(
        tender_id=td.get("tender_id",""),
        title=td.get("title",""),
        url=td.get("url",""),
        publisher=td.get("publisher",""),
        deadline=td.get("deadline",""),
        raw_metadata={"publish_date": td.get("publish_date",""), "update_date": td.get("update_date","")},
        pdf_text=td.get("pdf_text",""),
    )
    feedback = [f"תשובות הבהרה שהתקבלו:\n{answers_text}"]
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: analyze_tender(tender, settings, client, knowledge=knowledge, session_feedback=feedback)
    )
    _append_history(result, u)
    return result


# ── Generate clarification questions ──────────────────────────────────────────

@app.post("/api/tender/generate-questions")
async def generate_questions(body: dict, _: str = Depends(auth)):
    r = body
    client = _client()
    knowledge = _load_knowledge()
    prompt = f"""אתה יועץ עסקי של Electra Target. קראת את הניתוח הבא של מכרז ממשלתי.

כותרת: {r.get('title','')}
מפרסם: {r.get('publisher','')}
מועד הגשה: {r.get('deadline','')}
ניתוח: {r.get('analysis','')}

צור את כל שאלות ההבהרה שיש לשלוח למפרסם המכרז — כמה שצריך, עד 50 שאלות לכל היותר.
כלול כל שאלה חשובה שעולה מהמסמך. אל תגביל את עצמך למספר קבוע.

פלט כל שאלה בפורמט הבא (עמודה מופרדת ב-|):
מספר|עמוד|סעיף|שאלה

- עמוד: מספר עמוד רלוונטי במסמך אם ידוע, אחרת: כללי
- סעיף: מספר סעיף רלוונטי אם ידוע, אחרת: כללי
- שאלה: טקסט השאלה בעברית

דוגמה:
1|כללי|כללי|האם נדרש רישיון עסק?
2|5|3.2|מה תקופת האחריות על הציוד?

כתוב את כל השאלות בפורמט זה בלבד, ללא כותרות נוספות."""

    system = SYSTEM_PROMPT
    if knowledge:
        system += "\n\nתובנות שנצברו:\n" + "\n".join(f"- {g}" for g in knowledge)

    loop = asyncio.get_running_loop()
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-sonnet-4-6", max_tokens=4096, system=system,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        return {"questions": resp.content[0].text}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Export Excel (financial analysis) ─────────────────────────────────────────

@app.post("/api/tender/export-excel")
async def export_excel(body: dict, _: str = Depends(auth)):
    import io, re
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    r = body
    analysis = r.get("analysis", "")

    # Split analysis into sections
    fin_keys = ["הערכת היקף כספי", "ערך שנתי", "בסיס לחישוב", "כוח אדם נדרש", "אורך חוזה", "המלצה", "אתגרים", "סיכונים"]
    sections: dict[str, list[str]] = {}
    current_key = None
    for line in analysis.splitlines():
        clean = line.strip().replace("**","").replace("#","").strip()
        if not clean: continue
        matched = next((k for k in fin_keys if k in clean), None)
        if matched:
            current_key = matched
            sections.setdefault(current_key, [])
        if current_key:
            sections[current_key].append(clean)

    wb = Workbook()
    ws = wb.active
    ws.title = "ניתוח פיננסי"
    ws.sheet_view.rightToLeft = True

    hfill = PatternFill("solid", fgColor="1E3A5F")
    sfill = PatternFill("solid", fgColor="2563EB")
    thfill= PatternFill("solid", fgColor="374151")
    afill = PatternFill("solid", fgColor="EEF3FA")
    wfill = PatternFill("solid", fgColor="FFFFFF")
    thin  = Border(left=Side(style='thin',color='CCCCCC'), right=Side(style='thin',color='CCCCCC'),
                   top=Side(style='thin',color='CCCCCC'),  bottom=Side(style='thin',color='CCCCCC'))

    def is_md_table(line):
        return line.startswith('|') and line.endswith('|')

    def is_separator(line):
        return re.fullmatch(r'[\|\-\s:]+', line) is not None

    def parse_md_row(line):
        return [c.strip() for c in line.strip('|').split('|')]

    def rtl_align(horizontal="right", center=False):
        return Alignment(horizontal="center" if center else horizontal,
                         vertical="center", wrap_text=True)

    def style_cell(c, bold=False, fill=None, center=False):
        c.font = Font(bold=bold, name="Arial", size=10)
        if fill: c.fill = fill
        c.border = thin
        c.alignment = rtl_align(center=center)

    def hrow(label, row, fill=None, ncols=2):
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncols)
        c = ws.cell(row, 1, label)
        c.font = Font(bold=True, color="FFFFFF", size=12, name="Arial")
        c.fill = fill or hfill
        c.alignment = rtl_align()
        ws.row_dimensions[row].height = 24

    def drow(label, value, row, alt=False, ncols=2):
        fill = afill if alt else wfill
        c1 = ws.cell(row, 1, label or "")
        style_cell(c1, bold=bool(label), fill=fill)
        c2 = ws.cell(row, 2, str(value) if value else "")
        style_cell(c2, fill=fill)
        ws.row_dimensions[row].height = max(18, min(90, len(str(value or ""))//3+15))

    def write_section_lines(lines, start_row, max_cols):
        r = start_row
        alt = False
        i = 0
        while i < len(lines):
            line = lines[i]
            if is_md_table(line):
                # collect table block
                tbl_lines = []
                while i < len(lines) and (is_md_table(lines[i]) or is_separator(lines[i])):
                    tbl_lines.append(lines[i]); i += 1
                data_rows = [parse_md_row(l) for l in tbl_lines if not is_separator(l)]
                if not data_rows: continue
                ncols = max(len(row) for row in data_rows)
                # set/extend column widths
                for ci in range(1, ncols+1):
                    col_letter = ws.cell(r, ci).column_letter
                    ws.column_dimensions[col_letter].width = max(
                        ws.column_dimensions[col_letter].width, 18)
                for ri, dr in enumerate(data_rows):
                    is_hdr = ri == 0
                    row_fill = thfill if is_hdr else (afill if ri%2==0 else wfill)
                    for ci, val in enumerate(dr):
                        c = ws.cell(r, ci+1, val)
                        style_cell(c, bold=is_hdr, fill=row_fill,
                                   center=(ci < len(dr)-1))
                        if is_hdr: c.font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
                    ws.row_dimensions[r].height = 20
                    r += 1
            else:
                clean = line.replace("**","").replace("#","").strip()
                if clean and not is_separator(clean):
                    c = ws.cell(r, 1, clean)
                    style_cell(c, fill=(afill if alt else wfill))
                    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=max_cols)
                    ws.row_dimensions[r].height = max(18, min(60, len(clean)//5+15))
                    alt = not alt
                    r += 1
                i += 1
        return r

    # Auto-detect column count from tables in analysis
    max_cols = 2
    for line in analysis.splitlines():
        if is_md_table(line.strip()):
            max_cols = max(max_cols, line.count('|') - 1)
    max_cols = min(max_cols, 8)
    for ci in range(1, max_cols+1):
        letter = ws.cell(1, ci).column_letter
        ws.column_dimensions[letter].width = 20
    ws.column_dimensions[ws.cell(1,1).column_letter].width = 30

    row = 1
    hrow(f"ניתוח פיננסי: {r.get('title','')}", row, ncols=max_cols); row += 1
    for label, val, alt in [
        ("מפרסם", r.get("publisher",""), False),
        ("מועד הגשה", r.get("deadline",""), True),
        ("תאריך פרסום", r.get("publish_date",""), False),
    ]:
        drow(label, val, row, alt, ncols=max_cols); row += 1
    row += 1

    finance_order = ["הערכת היקף כספי", "ערך שנתי", "בסיס לחישוב", "כוח אדם נדרש", "אורך חוזה", "המלצה", "אתגרים", "סיכונים"]
    hrow("ניתוח פיננסי מפורט", row, sfill, ncols=max_cols); row += 1
    shown = set()
    for key in finance_order:
        lines = sections.get(key)
        if not lines or key in shown: continue
        shown.add(key)
        # Section sub-header
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=max_cols)
        c = ws.cell(row, 1, key)
        c.font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
        c.fill = PatternFill("solid", fgColor="2B5C9E")
        c.alignment = Alignment(horizontal="right", vertical="center")
        ws.row_dimensions[row].height = 20; row += 1
        row = write_section_lines(lines[1:], row, max_cols)  # skip key header line

    buf = io.BytesIO()
    wb.save(buf); buf.seek(0)
    tid = r.get("tender_id", "tender")
    return StreamingResponse(buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''%D7%A4%D7%99%D7%A0%D7%A0%D7%A1%D7%99_{tid}.xlsx"})


# ── Export Word (full analysis) ────────────────────────────────────────────────

@app.post("/api/tender/export-word")
async def export_word(body: dict, _: str = Depends(auth)):
    import io
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    r = body
    analysis  = r.get("analysis", "")
    questions = r.get("questions", "")
    knowledge = _load_knowledge()

    doc = Document()
    # Set document-level RTL
    sectPr = doc.sections[0]._sectPr
    sectPr.append(OxmlElement('w:bidi'))

    def set_rtl_para(par):
        par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        pPr = par._p.get_or_add_pPr()
        b = OxmlElement('w:bidi'); b.set(qn('w:val'), '1'); pPr.append(b)
        jc = OxmlElement('w:jc');  jc.set(qn('w:val'), 'right'); pPr.append(jc)

    def set_rtl_run(run):
        run.font.cs_name = "Arial"
        rPr = run._r.get_or_add_rPr()
        rtl = OxmlElement('w:rtl'); rtl.set(qn('w:val'), '1'); rPr.append(rtl)
        rPr.append(OxmlElement('w:cs'))

    def h(text, level=1):
        par = doc.add_heading(text, level=level)
        set_rtl_para(par)
        for run in par.runs:
            run.font.name = "Arial"
            set_rtl_run(run)
            if level == 1:
                run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x5F)
        return par

    from docx.shared import Cm
    from docx.oxml import OxmlElement as OE

    def add_run_inline(par, text):
        """Add run with inline **bold** parsing."""
        import re
        parts = re.split(r'\*\*(.+?)\*\*', text)
        for i, part in enumerate(parts):
            if not part: continue
            run = par.add_run(part)
            run.font.name = "Arial"; run.font.size = Pt(11)
            run.bold = (i % 2 == 1)
            set_rtl_run(run)

    def p(text, bold=False, bullet=False, size=11):
        par = doc.add_paragraph()
        set_rtl_para(par)
        if bullet:
            pPr = par._p.get_or_add_pPr()
            ind = OE('w:ind'); ind.set(qn('w:right'), '360'); pPr.append(ind)
        if bold:
            run = par.add_run(str(text))
            run.font.name = "Arial"; run.font.cs_name = "Arial"
            run.font.size = Pt(size); run.bold = True
            set_rtl_run(run)
        else:
            add_run_inline(par, str(text))
        return par

    def render_md(text):
        """Render markdown text into the Word document."""
        import re
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            # Heading
            hm = re.match(r'^(#{1,3})\s+(.*)', stripped)
            if hm:
                level = min(len(hm.group(1)) + 1, 3)
                h(hm.group(2).replace('**',''), level)
                i += 1; continue
            # Table
            if stripped.startswith('|') and stripped.endswith('|'):
                tbl_lines = []
                while i < len(lines) and lines[i].strip().startswith('|'):
                    tbl_lines.append(lines[i].strip()); i += 1
                data_rows = [r for r in tbl_lines if not re.fullmatch(r'[\|\-\s:]+', r)]
                if not data_rows: continue
                parsed = [[c.strip() for c in r.strip('|').split('|')] for r in data_rows]
                ncols = max(len(r) for r in parsed)
                tbl = doc.add_table(rows=len(parsed), cols=ncols)
                tbl.style = 'Table Grid'
                for ri, row in enumerate(parsed):
                    for ci, val in enumerate(row):
                        cell = tbl.rows[ri].cells[ci]
                        cell.paragraphs[0].clear()
                        cp = cell.paragraphs[0]
                        cp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                        pPr = cp._p.get_or_add_pPr()
                        b = OE('w:bidi'); b.set(qn('w:val'),'1'); pPr.append(b)
                        run = cp.add_run(val)
                        run.font.name = "Arial"; run.font.size = Pt(10)
                        run.bold = (ri == 0)
                        if ri == 0: run.font.color.rgb = RGBColor(0xFF,0xFF,0xFF)
                        set_rtl_run(run)
                        tcPr = cell._tc.get_or_add_tcPr()
                        shd = OE('w:shd')
                        shd.set(qn('w:fill'), '1E3A5F' if ri==0 else ('EEF3FA' if ri%2==0 else 'FFFFFF'))
                        shd.set(qn('w:color'),'auto'); shd.set(qn('w:val'),'clear')
                        tcPr.append(shd)
                continue
            # Bullet
            bm = re.match(r'^[-*+]\s+(.*)', stripped)
            if bm:
                p(f"• {bm.group(1)}", bullet=True); i += 1; continue
            # Numbered list
            nm = re.match(r'^\d+\.\s+(.*)', stripped)
            if nm:
                p(f"• {nm.group(1)}", bullet=True); i += 1; continue
            # Horizontal rule or separator — skip
            if re.fullmatch(r'[-_*]{2,}', stripped):
                i += 1; continue
            # Empty
            if not stripped:
                i += 1; continue
            # Normal
            p(stripped)
            i += 1

    h(f"ניתוח מכרז: {r.get('title','')}", 1)
    if r.get('publisher'):    p(f"מפרסם: {r['publisher']}")
    if r.get('publish_date'): p(f"תאריך פרסום: {r['publish_date']}")
    if r.get('deadline'):     p(f"מועד הגשה: {r['deadline']}")
    if r.get('update_date'):  p(f"תאריך עדכון: {r['update_date']}")
    doc.add_paragraph()

    render_md(analysis.replace('<','').replace('>',''))

    buf = io.BytesIO()
    doc.save(buf); buf.seek(0)
    tid = r.get("tender_id", "tender")
    return StreamingResponse(buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''%D7%A0%D7%99%D7%AA%D7%95%D7%97_{tid}.docx"})


# ── Export questions to Word (table) ──────────────────────────────────────────

@app.post("/api/tender/export-questions-word")
async def export_questions_word(body: dict, _: str = Depends(auth)):
    import io
    from docx import Document
    from docx.shared import Pt, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    r = body
    questions_raw = r.get("questions", "")

    doc = Document()
    sectPr = doc.sections[0]._sectPr
    sectPr.append(OxmlElement('w:bidi'))

    def set_rtl(par):
        pPr = par._p.get_or_add_pPr()
        b = OxmlElement('w:bidi'); b.set(qn('w:val'), '1'); pPr.append(b)
        jc = OxmlElement('w:jc');  jc.set(qn('w:val'), 'right'); pPr.append(jc)

    def set_rtl_run(run):
        rPr = run._r.get_or_add_rPr()
        rtl = OxmlElement('w:rtl'); rtl.set(qn('w:val'), '1'); rPr.append(rtl)
        rPr.append(OxmlElement('w:cs'))
        run.font.cs_name = "Arial"

    title_p = doc.add_heading(f"שאלות הבהרה: {r.get('title','')}", 1)
    set_rtl(title_p)
    for run in title_p.runs:
        run.font.name = "Arial"; run.font.cs_name = "Arial"
        run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x5F)
        set_rtl_run(run)

    def meta_line(text):
        p = doc.add_paragraph(); set_rtl(p)
        run = p.add_run(text)
        run.font.name = "Arial"; run.font.cs_name = "Arial"; run.font.size = Pt(11)
        set_rtl_run(run)

    if r.get('publisher'):    meta_line(f"מפרסם: {r['publisher']}")
    if r.get('publish_date'): meta_line(f"תאריך פרסום: {r['publish_date']}")
    if r.get('deadline'):     meta_line(f"מועד הגשה: {r['deadline']}")
    if r.get('update_date'):  meta_line(f"תאריך עדכון: {r['update_date']}")
    doc.add_paragraph()

    # Parse pipe-separated questions
    rows = []
    for line in questions_raw.strip().splitlines():
        line = line.strip()
        if not line: continue
        parts = line.split("|")
        if len(parts) >= 4:
            rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip(), "|".join(parts[3:]).strip()))
        elif line:
            num = str(len(rows)+1)
            rows.append((num, "כללי", "כללי", line.lstrip("0123456789. ")))

    # Column order as written in file — RTL doc renders col1 on the right
    # so מספר(col1) | עמוד(col2) | סעיף(col3) | שאלה(col4) displays correctly R→L
    headers_rtl  = ["שאלה", "סעיף", "עמוד", "מספר"]
    col_widths_rtl = [Cm(11), Cm(2.5), Cm(2), Cm(1.5)]

    def add_shd(tc, color):
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:fill'), color); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:val'), 'clear')
        tcPr.append(shd)

    def cell_rtl_run(cell, text, bold=False, size=10, color=None):
        p = cell.paragraphs[0]; p.clear()
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        pPr = p._p.get_or_add_pPr()
        b = OxmlElement('w:bidi'); b.set(qn('w:val'),'1'); pPr.append(b)
        jc = OxmlElement('w:jc'); jc.set(qn('w:val'),'right'); pPr.append(jc)
        run = p.add_run(text)
        run.font.name = "Arial"; run.font.cs_name = "Arial"; run.font.size = Pt(size); run.bold = bold
        if color: run.font.color.rgb = color
        rPr = run._r.get_or_add_rPr()
        rtl = OxmlElement('w:rtl'); rtl.set(qn('w:val'),'1'); rPr.append(rtl)
        rPr.append(OxmlElement('w:cs'))

    table = doc.add_table(rows=1+len(rows), cols=4)
    table.style = "Table Grid"

    # Header row
    hdr = table.rows[0]
    for i, (h_text, w) in enumerate(zip(headers_rtl, col_widths_rtl)):
        cell = hdr.cells[i]; cell.width = w
        add_shd(cell._tc, '1E3A5F')
        cell_rtl_run(cell, h_text, bold=True, size=11, color=RGBColor(0xFF,0xFF,0xFF))

    # Data rows
    for ri, (num, page, section, question) in enumerate(rows):
        row_cells = table.rows[ri+1].cells
        fill_color = 'EEF3FA' if ri % 2 == 0 else 'FFFFFF'
        for ci, text in enumerate([question, section, page, num]):
            cell = row_cells[ci]
            add_shd(cell._tc, fill_color)
            cell_rtl_run(cell, text, size=10)

    buf = io.BytesIO()
    doc.save(buf); buf.seek(0)
    tid = r.get("tender_id", "tender")
    return StreamingResponse(buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''%D7%A9%D7%90%D7%9C%D7%95%D7%AA_{tid}.docx"})


# ── Email digest ──────────────────────────────────────────────────────────────

@app.post("/api/send-email")
async def send_email_digest(body: dict, u: str = Depends(auth)):
    from emailer import send_digest, build_html_digest
    from datetime import date
    import re

    tenders   = body.get("tenders", [])   # list of tender result dicts
    min_level = body.get("min_level", "high")  # "high", "medium", "low"

    level_rank = {"high": 0, "medium": 1, "low": 2}
    threshold  = level_rank.get(min_level, 0)

    def get_rank(analysis):
        if "גבוהה" in analysis: return 0
        if "בינונית" in analysis: return 1
        return 2

    def extract_summary(analysis: str) -> str:
        """Extract the first סיכום section from analysis."""
        lines = analysis.splitlines()
        in_section = False
        result = []
        for line in lines:
            if re.search(r'סיכום', line):
                in_section = True
                continue
            if in_section:
                if re.match(r'^#{1,3}\s', line) and result:
                    break
                if line.strip():
                    result.append(line.strip().replace('**','').replace('#',''))
        return ' '.join(result[:5]) if result else analysis[:300]

    filtered = [t for t in tenders if get_rank(t.get("analysis","")) <= threshold]
    if not filtered:
        return {"ok": False, "msg": "לא נמצאו מכרזים ברמה שנבחרה"}

    app_url = "https://tender-scanner.up.railway.app"
    def badge(a):
        if "גבוהה" in a: return "🟢 גבוהה"
        if "בינונית" in a: return "🟡 בינונית"
        return "🔴 נמוכה"
    def badge_color(a):
        if "גבוהה" in a: return "#1a7a1a"
        if "בינונית" in a: return "#b36b00"
        return "#8b0000"

    cards = []
    for t in filtered:
        analysis = t.get("analysis","")
        summary  = extract_summary(analysis)
        color    = badge_color(analysis)
        cards.append(f"""
        <div style="border:1px solid #ddd;border-radius:8px;padding:16px;margin-bottom:20px;font-family:Arial,sans-serif;direction:rtl;text-align:right;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <h2 style="margin:0;font-size:16px;color:{color}">{t.get('title','')}</h2>
            <span style="background:{color};color:#fff;padding:3px 10px;border-radius:12px;font-size:12px;white-space:nowrap">{badge(analysis)}</span>
          </div>
          <p style="margin:4px 0;color:#555;font-size:13px">
            {t.get('publisher','') or ''}
            {' | פרסום: '+t['publish_date'] if t.get('publish_date') else ''}
            {' | הגשה: '+t['deadline'] if t.get('deadline') else ''}
          </p>
          <hr style="border:none;border-top:1px solid #eee;margin:10px 0">
          <p style="font-size:14px;line-height:1.7;margin:0 0 12px">{summary}</p>
          <div style="display:flex;gap:12px">
            <a href="{t.get('url','')}" style="color:#2563EB;font-size:13px">🔗 עמוד המכרז</a>
            <a href="{app_url}" style="color:#2563EB;font-size:13px">📊 ניתוח מלא במערכת</a>
          </div>
        </div>""")

    run_date = date.today().strftime("%d/%m/%Y")
    level_label = {"high":"גבוהה בלבד","medium":"בינונית וגבוהה","low":"כל הרמות"}.get(min_level,"")
    html = f"""<html><body style="background:#f5f5f5;padding:20px">
      <h1 style="font-family:Arial,sans-serif;direction:rtl;text-align:right;color:#1a1a2e">
        סריקת מכרזים — {run_date}
      </h1>
      <p style="font-family:Arial,sans-serif;direction:rtl;text-align:right;color:#555">
        {len(filtered)} מכרזים ברמה: {level_label}
      </p>
      {''.join(cards)}
      <p style="font-family:Arial,sans-serif;font-size:12px;color:#999;text-align:center;margin-top:30px">Electra Target Tools</p>
    </body></html>"""

    settings   = _load_settings()
    app_pw     = os.environ.get("GMAIL_APP_PASSWORD","")
    if not app_pw:
        return {"ok": False, "msg": "GMAIL_APP_PASSWORD לא מוגדר בסביבה"}

    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import smtplib

    sender    = settings["email"]["sender"]
    recipient = settings["email"]["recipient"]
    subject   = f"[Electra Target] {len(filtered)} מכרזים — {run_date}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, app_pw)
            server.sendmail(sender, [recipient], msg.as_string())
        return {"ok": True, "count": len(filtered)}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


# ── Learning mode ─────────────────────────────────────────────────────────────

@app.post("/api/learn/analyze")
async def learn_analyze(body: dict, u: str = Depends(auth)):
    url = body.get("url", "").strip()
    settings = _load_settings()
    tid = _extract_id_from_url(url)
    meta = {"tender_id": tid, "title": url, "url": url, "publish_date": "", "update_date": ""}
    tender = await asyncio.wait_for(fetch_tender_detail(meta, settings), timeout=90)
    if not tender.title or tender.title.startswith("http"):
        tender.title = f"מכרז {tid}"
    td = {
        "tender_id": tender.tender_id, "title": tender.title, "url": tender.url,
        "publisher": tender.publisher, "deadline": tender.deadline,
        "pdf_text": tender.pdf_text, "has_pdf": bool(tender.pdf_text),
    }
    if not tender.pdf_text:
        result = {"title": tender.title, "url": url, "publisher": tender.publisher,
                  "deadline": tender.deadline, "has_pdf": False, "analysis": "NO_PDF"}
    else:
        client = _client()
        loop = asyncio.get_running_loop()
        knowledge = _load_knowledge()
        result = await loop.run_in_executor(
            None, lambda t=tender: analyze_tender(t, settings, client, knowledge=knowledge)
        )
    return {"tender": td, "result": result}

@app.post("/api/learn/reanalyze")
async def learn_reanalyze(body: dict, _: str = Depends(auth)):
    td = body.get("tender", {})
    feedback = body.get("feedback", [])
    settings = _load_settings()
    tender_obj = Tender(
        tender_id=td["tender_id"], title=td["title"], url=td["url"],
        publisher=td.get("publisher",""), deadline=td.get("deadline",""),
        pdf_text=td.get("pdf_text",""), raw_metadata={"publish_date":"","update_date":""},
    )
    client = _client()
    loop = asyncio.get_running_loop()
    knowledge = _load_knowledge()
    result = await loop.run_in_executor(
        None,
        lambda: analyze_tender(tender_obj, settings, client, knowledge=knowledge, session_feedback=feedback)
    )
    return result

@app.post("/api/learn/save")
async def learn_save(body: dict, _: str = Depends(auth)):
    title = body.get("title", "")
    feedback = body.get("feedback", [])
    client = _client()
    loop = asyncio.get_running_loop()
    new_g = await loop.run_in_executor(None, lambda: distill_knowledge(title, feedback, client))
    existing = _load_knowledge()
    existing.extend(new_g)
    _save_knowledge(existing)
    return {"new": new_g, "total": len(existing)}


# ── Shift comparison ──────────────────────────────────────────────────────────

_DEFAULT_SHIFTS_CFG = {
    "excel_columns": {"date": "B", "start_time": "C", "end_time": "D", "worker_name": "F"},
    "excel_has_header": True,
    "rules": {"gap_threshold_minutes": 30, "default_start_time": "10:00"},
    "aliases": {},
    "managers": [],
    "ignored_names": [],
}

def _shifts_cfg_path(u: str) -> Path:
    return _udir(u) / "shifts_config.json"

def _load_shifts_cfg(u: str) -> dict:
    return _rj(_shifts_cfg_path(u), dict(_DEFAULT_SHIFTS_CFG))

def _save_shifts_cfg(u: str, cfg: dict):
    _wj(_shifts_cfg_path(u), cfg)

@app.get("/api/shifts/config")
async def shifts_config(u: str = Depends(auth)):
    return _load_shifts_cfg(u)

@app.post("/api/shifts/config")
async def save_shifts_config(body: dict, u: str = Depends(auth)):
    _save_shifts_cfg(u, body)
    return {"ok": True}

@app.post("/api/shifts/validate")
async def shifts_validate(_: str = Depends(auth), message: str = Form(...)):
    return {"suspicious": find_suspicious_lines(message)}

@app.post("/api/shifts/compare")
async def shifts_compare(
    u: str = Depends(auth),
    message: str = Form(...),
    excel: UploadFile = File(...),
):
    cfg = _load_shifts_cfg(u)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(await excel.read())
        tmp_path = Path(tmp.name)
    try:
        msg_entries = parse_message(message)
        excel_entries = parse_excel(str(tmp_path), cfg)
        results = compare(msg_entries, excel_entries, cfg)

        out_path = Path(tempfile.mktemp(suffix=".xlsx"))
        export_to_excel(results, str(out_path))
        excel_bytes = out_path.read_bytes()
        out_path.unlink(missing_ok=True)

        token = secrets.token_urlsafe(16)
        _excel_cache[token] = excel_bytes

        def _f(v):
            if hasattr(v, "strftime"):
                return v.strftime("%H:%M") if hasattr(v, "hour") else v.strftime("%d/%m/%Y")
            return str(v) if v else ""

        return {
            "results": [
                {
                    "status": r["status"],
                    "worker_name": r.get("worker_name", ""),
                    "workplace": r.get("workplace", ""),
                    "date": _f(r.get("date")),
                    "start_time": _f(r.get("start_time")),
                    "end_time": _f(r.get("end_time")),
                    "sales": str(r.get("sales", "")) if r.get("sales") else "",
                    "notes": r.get("notes", ""),
                }
                for r in results
            ],
            "token": token,
        }
    finally:
        tmp_path.unlink(missing_ok=True)

@app.get("/api/shifts/download/{token}")
async def shifts_download(token: str, _: str = Depends(auth)):
    data = _excel_cache.get(token)
    if not data:
        raise HTTPException(404, "קובץ לא נמצא")
    return Response(content=data,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": "attachment; filename=comparison.xlsx"})


# ── Recruiter Analysis ────────────────────────────────────────────────────────

RECRUITER_CONFIG_FILE = BASE_DIR / "data" / "shared" / "recruiter_config.json"
RECRUITER_DATA_FILE = BASE_DIR / "data" / "shared" / "recruiter_data.json"

_DEFAULT_RECRUITER_CONFIG: dict = {
    "recruiters": [],
    "long_call_threshold_minutes": 8,
    "repeat_call_threshold": 2,
    "default_days_back": 7,
}

def _load_recruiter_config() -> dict:
    return _rj(RECRUITER_CONFIG_FILE, dict(_DEFAULT_RECRUITER_CONFIG))

def _save_recruiter_config(cfg: dict):
    _wj(RECRUITER_CONFIG_FILE, cfg)

@app.get("/api/recruiter/config")
async def recruiter_config_get(_: str = Depends(auth)):
    return _load_recruiter_config()

@app.post("/api/recruiter/config")
async def recruiter_config_post(body: dict, _: str = Depends(auth)):
    _save_recruiter_config(body)
    return {"ok": True}

@app.post("/api/recruiter/upload")
async def recruiter_upload(excel: UploadFile = File(...), _: str = Depends(auth)):
    import pandas as pd
    from datetime import datetime as _dt

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(await excel.read())
        tmp_path = Path(tmp.name)

    try:
        fname = excel.filename or ""
        if fname.lower().endswith(".csv"):
            df = pd.read_csv(str(tmp_path))
        else:
            df = pd.read_excel(str(tmp_path), sheet_name="פיד", header=0)
    finally:
        tmp_path.unlink(missing_ok=True)

    def _ext(val) -> str | None:
        try:
            s = str(int(float(val)))
            if s.startswith("910") and len(s) >= 6:
                return s[3:].lstrip("0") or s[3:]
            if 2 <= len(s) <= 4:
                return s.lstrip("0") or s
        except Exception:
            pass
        return None

    calls = []
    for _, row in df.iterrows():
        ext = _ext(row.get("src"))
        if not ext:
            continue
        raw_date = row.get("calldate")
        try:
            if pd.isna(raw_date):
                continue
        except Exception:
            if not raw_date:
                continue
        try:
            dt = pd.to_datetime(raw_date)
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M")
            hour = int(dt.hour)
        except Exception:
            continue
        try:
            duration = int(float(row.get("billsec") or 0))
        except Exception:
            duration = 0
        answered = str(row.get("disposition", "")).upper() == "ANSWERED"
        dst = str(row.get("dst", "")).rstrip(".0").strip()
        calls.append({
            "date": date_str, "time": time_str, "hour": hour,
            "extension": ext, "dst": dst,
            "duration_sec": duration, "answered": answered,
        })

    _wj(RECRUITER_DATA_FILE, {"last_updated": _dt.now().isoformat(), "calls": calls})
    return {"ok": True, "total_calls": len(calls)}


@app.get("/api/recruiter/data")
async def recruiter_data(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    _: str = Depends(auth),
):
    import collections

    raw = _rj(RECRUITER_DATA_FILE, None)
    if not raw:
        return {"last_updated": None, "recruiters": [], "all_dates": [], "full_range": None}

    cfg = _load_recruiter_config()
    recruiter_map = {r["extension"]: r["name"] for r in cfg.get("recruiters", [])}
    long_sec = cfg.get("long_call_threshold_minutes", 8) * 60
    repeat_min = cfg.get("repeat_call_threshold", 2)

    all_calls = raw.get("calls", [])
    all_call_dates = sorted(set(c["date"] for c in all_calls))
    full_range = {"from": all_call_dates[0], "to": all_call_dates[-1]} if all_call_dates else None

    df = date_from or (all_call_dates[0] if all_call_dates else "")
    dt = date_to or (all_call_dates[-1] if all_call_dates else "")
    calls = [c for c in all_calls if df <= c["date"] <= dt]
    all_dates = sorted(set(c["date"] for c in calls))

    by_ext: dict = collections.defaultdict(list)
    for c in calls:
        by_ext[c["extension"]].append(c)

    recruiters = []
    for ext, rcalls in by_ext.items():
        name = recruiter_map.get(ext, f"שלוחה {ext}")
        answered = [c for c in rcalls if c["answered"]]
        total = len(rcalls)
        ans_count = len(answered)
        ans_rate = round(ans_count / total * 100, 1) if total else 0
        total_sec = sum(c["duration_sec"] for c in answered)
        total_min = round(total_sec / 60, 1)
        avg_dur = round(total_sec / ans_count / 60, 1) if ans_count else 0
        work_days = len(set(c["date"] for c in rcalls))
        avg_per_day = round(total / work_days, 1) if work_days else 0
        times = sorted(c["time"] for c in rcalls)

        hourly: dict = collections.defaultdict(float)
        for c in answered:
            hourly[c["hour"]] += c["duration_sec"] / 60
        hourly_dist = {str(h): round(hourly.get(h, 0), 1) for h in range(8, 18)}

        daily_calls: dict = collections.defaultdict(int)
        daily_minutes: dict = collections.defaultdict(float)
        for c in rcalls:
            daily_calls[c["date"]] += 1
        for c in answered:
            daily_minutes[c["date"]] += c["duration_sec"] / 60

        long_calls = sorted(
            [{"date": c["date"], "time": c["time"], "dst": c["dst"],
              "minutes": round(c["duration_sec"] / 60, 1)}
             for c in answered if c["duration_sec"] >= long_sec],
            key=lambda x: x["minutes"], reverse=True,
        )

        dst_counts = collections.Counter(c["dst"] for c in rcalls if c["dst"])
        repeat_numbers = [
            {"number": num, "count": cnt}
            for num, cnt in dst_counts.most_common(20)
            if cnt >= repeat_min
        ]

        recruiters.append({
            "name": name, "extension": ext,
            "total_calls": total, "answered_calls": ans_count, "answer_rate": ans_rate,
            "total_minutes": total_min, "avg_duration_minutes": avg_dur,
            "work_days": work_days, "avg_calls_per_day": avg_per_day,
            "first_call": times[0] if times else None,
            "last_call": times[-1] if times else None,
            "hourly_distribution": hourly_dist,
            "daily_calls": dict(daily_calls),
            "daily_minutes": {d: round(v, 1) for d, v in daily_minutes.items()},
            "long_calls": long_calls,
            "repeat_numbers": repeat_numbers,
        })

    known_order = {r["extension"]: i for i, r in enumerate(cfg.get("recruiters", []))}
    recruiters.sort(key=lambda r: known_order.get(r["extension"], 999))

    return {"last_updated": raw.get("last_updated"), "all_dates": all_dates, "full_range": full_range, "recruiters": recruiters}
