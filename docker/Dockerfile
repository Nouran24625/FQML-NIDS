# ---------- stage 1: compile liboqs ----------
FROM python:3.11-slim AS liboqs-build
ARG LIBOQS_VERSION=0.16.0
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 -b ${LIBOQS_VERSION} https://github.com/open-quantum-safe/liboqs /src/liboqs \
    && cmake -S /src/liboqs -B /src/liboqs/build -GNinja \
    -DBUILD_SHARED_LIBS=ON -DOQS_BUILD_ONLY_LIB=ON \
    -DCMAKE_INSTALL_PREFIX=/opt/liboqs \
    -DOQS_MINIMAL_BUILD="KEM_ml_kem_512;KEM_ml_kem_768;KEM_ml_kem_1024;SIG_ml_dsa_44;SIG_ml_dsa_65;SIG_ml_dsa_87" \
    && ninja -C /src/liboqs/build install

# ---------- stage 2: the image we actually use ----------
FROM python:3.11-slim
ARG LIBOQS_PYTHON_VERSION=0.16.0.1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OQS_INSTALL_PATH=/opt/liboqs LD_LIBRARY_PATH=/opt/liboqs/lib

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=liboqs-build /opt/liboqs /opt/liboqs

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt \
    && pip install "git+https://github.com/open-quantum-safe/liboqs-python@${LIBOQS_PYTHON_VERSION}"

CMD ["bash"]