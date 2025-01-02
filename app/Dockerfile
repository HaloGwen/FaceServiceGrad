FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED 1
ENV TZ=Asia/Ho_Chi_Minh
ENV WORKERS=10

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt update -y && apt install -y build-essential default-libmysqlclient-dev && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN mkdir -p /src/logs

FROM base AS dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS final
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin
COPY . /src/
RUN chmod +x /src/run_server.sh

CMD ["./run_server.sh"]
