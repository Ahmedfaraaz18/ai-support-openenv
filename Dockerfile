FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir pydantic openai flask pyyaml

EXPOSE 7860

ENV FLASK_APP=scripts/server.py
ENV FLASK_ENV=production

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=7860"]
