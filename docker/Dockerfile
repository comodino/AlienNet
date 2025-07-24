FROM jupyter/scipy-notebook:2023-10-20

USER root
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
USER jovyan

WORKDIR /home/jovyan/work

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
