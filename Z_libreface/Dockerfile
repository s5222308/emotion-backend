FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- System deps ----
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps ----
COPY requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir libreface python-multipart

# If fastapi/uvicorn/pandas ARE NOT in requirements.txt,
# then do this instead of the block above:
# RUN pip3 install --no-cache-dir --upgrade pip && \
#     pip3 install --no-cache-dir fastapi "uvicorn[standard]" pandas libreface

# ---- App code ----
COPY . /app

# Match the actual port uvicorn will use
EXPOSE 9001

# Run uvicorn directly, app is defined in app.py as `app = FastAPI(...)`
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9001"]
