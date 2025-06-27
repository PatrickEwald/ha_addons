FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential cmake libomp-dev jq

ENV MAKEFLAGS="-j$(nproc)"

RUN pip install --no-cache-dir lightgbm scikit-learn numpy pandas requests pytz holidays influxdb==5.3.1

COPY forecastOP.py /opt/forecast/forecastOP.py
COPY rootfs/usr/local/bin/run.sh /usr/local/bin/run.sh
RUN chmod +x /usr/local/bin/run.sh

CMD ["/usr/local/bin/run.sh"]