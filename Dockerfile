FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

COPY bioimpedance_dx/ ./bioimpedance_dx/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "bioimpedance_dx.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
