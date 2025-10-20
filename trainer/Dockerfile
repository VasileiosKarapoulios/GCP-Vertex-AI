FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Stockholm

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 tmux tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

COPY . /app
COPY requirements.txt .

COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "-m", "train"]



