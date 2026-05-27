FROM python:3.11-slim

# ffmpeg: used for the duration probe and the first-solid energy decode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch + torchaudio from the CPU wheel index — the default PyPI index
# pulls the multi-GB CUDA build, which this CPU-only sidecar never uses.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchaudio

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Bake the checkpoint into the image so a cold container starts (and
# runs offline) without a first-request download.
RUN python -c "from beat_this.inference import Audio2Beats; Audio2Beats(checkpoint_path='final0', device='cpu', dbn=False)"

ENV MUSIC_ROOT=/music \
    BEATTHIS_CHECKPOINT=final0 \
    BEATTHIS_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
