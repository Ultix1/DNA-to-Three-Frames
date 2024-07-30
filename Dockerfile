FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04

WORKDIR /app

COPY . .

# Needed to add PPA
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y --no-install-recommends software-properties-common

# Packages in requirements.txt only work in Python 3.9
RUN apt update && add-apt-repository ppa:deadsnakes/ppa && apt install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3.9-venv

RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN python -m pip install -r requirements.txt

CMD ["python", "-c", "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"]