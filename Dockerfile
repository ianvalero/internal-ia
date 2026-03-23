FROM python:3.11-slim

WORKDIR /app

ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY

COPY cerebro/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cerebro ./cerebro

EXPOSE 9000

CMD ["uvicorn", "cerebro.main:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]