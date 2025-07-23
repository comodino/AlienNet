FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# System packages
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Per Jupyter
RUN pip install jupyterlab ipywidgets

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]