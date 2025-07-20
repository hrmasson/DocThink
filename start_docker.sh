#!/bin/bash
set -e

if [ -f .env ]; then
  export $(cat .env | xargs)
fi

MODEL_NAME="mistral"
MODEL_FILE="mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/$MODEL_FILE"
MODEL_DIR="models/$MODEL_NAME"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"


# 1. Ensure model exists on host
if [ ! -f "$MODEL_PATH" ]; then
  echo "🔽 Model not found locally. Downloading to $MODEL_PATH..."
  mkdir -p "$MODEL_DIR"
  curl -L "$MODEL_URL" -o "$MODEL_PATH"
else
  echo "✅ Model already exists at $MODEL_PATH"
fi

# 2. Start Docker
echo "🚀 Starting Docker containers..."
docker-compose up -d

# 3. Wait for LocalAI API to become available
echo "⏳ Waiting for LocalAI to be ready on port ${LOCALAI_PORT}..."
until curl -s http://localhost:${LOCALAI_PORT}/v1/models > /dev/null; do
    echo "⌛ Still waiting for LocalAI..."
    sleep 5
done

# 4. Wait for the model to appear
echo "⏳ Waiting for the model '$MODEL_NAME' to load..."
until curl -s http://localhost:${LOCALAI_PORT}/v1/models | grep -q "$MODEL_NAME"; do
    echo "⌛ Model is still loading..."
    sleep 15
done

echo "✅ Model '$MODEL_NAME' is ready and accessible at http://localhost:${LOCALAI_PORT}"
