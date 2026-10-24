FROM python:3.11-slim-bookworm

WORKDIR /app

COPY serve/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY src /app/src
COPY serve/app.py serve/inference.py serve/checkpoint.pt /app/serve/

ENV PYTHONPATH=/app/src
ENV HUMANIZE_CHECKPOINT=/app/serve/checkpoint.pt

EXPOSE 7860
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "7860"]
