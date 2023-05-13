FROM python:3.8

# ARG UID
# ARG GID

WORKDIR /app

# RUN groupadd -o --gid $GID myuser && useradd -m -u $UID --gid $GID myuser && chown -R myuser /app

COPY requirements.txt .
COPY api.py .

RUN pip --no-cache-dir install -r requirements.txt

# ENV PYTHONPATH "${PYTHONPATH}:/app/resources"
# USER myuser

EXPOSE 8501

# CMD ["python", "./api.py"]
CMD ["streamlit", "run", "--server.port", "8501", "--server.enableCORS", "false", "api.py"]