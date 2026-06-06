FROM python:3.11-slim

WORKDIR /app

# system deps occasionally needed by tokenizers / sentencepiece wheels
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# caches must live in a writable location on HF Spaces
ENV HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py"]
