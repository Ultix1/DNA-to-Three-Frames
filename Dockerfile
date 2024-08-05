FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04

WORKDIR /app

COPY . .

# Needed to add PPA
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y --no-install-recommends software-properties-common

# Packages in requirements.txt only work in Python 3.9
RUN apt update && add-apt-repository ppa:deadsnakes/ppa && apt install -y \
    curl \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3.9-venv

RUN curl -Lo /tmp/blast.tgz https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.16.0+-x64-linux.tar.gz && \
    tar -xzvf /tmp/blast.tgz -C /usr/local/
RUN curl -Lo /usr/local/bin/clustalo http://www.clustal.org/omega/clustalo-1.2.4-Ubuntu-x86_64 && chmod u+x /usr/local/bin/clustalo
ENV PATH="/usr/local/ncbi-blast-2.16.0+/bin:$PATH"

RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/pip \
	--mount=type=bind,source=requirements.txt,target=requirements.txt \
	python -m pip install -r requirements.txt

CMD ["python", "-c", "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"]
