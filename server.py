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

from analyzer import analyze_tender, distill_knowledge
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

@app.get("/api/scan")
async def scan(u: str = Depends(auth), skip_seen: bool = Query(True)):
    settings = _load_settings()

    async def gen():
        client = _client()
        loop = asyncio.get_running_loop()

        yield f"data: {json.dumps({'type':'status','msg':'מתחבר ל-mr.gov.il...'})}\n\n"
        try:
            tender_list = await asyncio.wait_for(fetch_tender_list(settings), timeout=120)
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','msg':str(e)})}\n\n"
            return

        seen_path = _udir(u) / "seen.json"
        seen = load_seen(seen_path) if skip_seen else set()
        new = filter_new(tender_list, seen) if skip_seen else tender_list

        yield f"data: {json.dumps({'type':'count','total':len(tender_list),'new':len(new)})}\n\n"

        if not new:
            yield f"data: {json.dumps({'type':'complete','count':0})}\n\n"
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

        yield f"data: {json.dumps({'type':'complete','count':len(results)})}\n\n"

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
