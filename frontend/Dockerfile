FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV OLLAMA_BASE=http://ollama:11434/v1
ENV OLLAMA_KEY=ollama
ENV OLLAMA_MODEL=qwen3:0.6b
ENV BACKEND_URL=http://backend:8000
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
