FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-heb poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/shared data/users

ARG ANTHROPIC_API_KEY
ENV ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
ARG GMAIL_APP_PASSWORD
ENV GMAIL_APP_PASSWORD=$GMAIL_APP_PASSWORD

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/api/me || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8501"]
