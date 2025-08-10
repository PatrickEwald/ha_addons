ARG BUILD_FROM
FROM ${BUILD_FROM}

RUN apk add --no-cache python3

COPY app /app
COPY rootfs/ /

RUN chmod +x /etc/services.d/amazon/run
