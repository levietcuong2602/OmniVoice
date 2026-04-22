FROM --platform=linux/amd64 pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install package dependencies first (layer caching)
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir \
    "torch==2.8.0" \
    "torchaudio==2.8.0" \
    --index-url https://download.pytorch.org/whl/cu128

# Install the rest of the Python deps
COPY omnivoice/ ./omnivoice/
RUN pip install --no-cache-dir -e .

# HuggingFace cache lives in a volume
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface/hub

EXPOSE 7860

ENTRYPOINT ["omnivoice-demo"]
CMD ["--model", "k2-fsa/OmniVoice", "--port", "7860", "--ip", "0.0.0.0"]
