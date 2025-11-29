FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenFace
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    python3 \
    python3-dev \
    python3-pip \
    libopencv-dev \
    libboost-all-dev \
    libtbb-dev \
    libopenblas-dev \
    libdlib-dev \
    && rm -rf /var/lib/apt/lists/*

# Build dlib from source
RUN git clone https://github.com/davisking/dlib.git /opt/dlib && \
    cd /opt/dlib && \
    git checkout v19.24 && \
    mkdir build && cd build && \
    cmake .. && \
    cmake --build . --config Release && \
    make install && \
    ldconfig

# Clone and build OpenFace
RUN git clone https://github.com/TadasBaltrusaitis/OpenFace.git /opt/OpenFace
WORKDIR /opt/OpenFace
RUN bash ./download_models.sh
RUN mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE .. && \
    make -j$(nproc)

# Set OpenFace in path
ENV PATH="/opt/OpenFace/build/bin:${PATH}"

# Install Flask
RUN pip3 install flask

# Set up app directory
WORKDIR /app
COPY app.py /app/app.py

EXPOSE 9000

CMD ["python3", "app.py"]