FROM python:3.9-slim

WORKDIR /app

# Cài đặt ntpdate để đồng bộ thời gian
RUN apt-get update && apt-get install -y ntpdate \
    && ntpdate pool.ntp.org \
    && apt-get remove -y ntpdate \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Cập nhật pip
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "main2.py"]