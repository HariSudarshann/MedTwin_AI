#!/usr/bin/env bash
# MediTwin AI — Quick Run Script
# Usage: ./run.sh [streamlit|backend|train|test]

set -e
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || echo "No venv found — using system Python"

CMD=${1:-streamlit}

case "$CMD" in
  streamlit)
    echo "🚀 Starting MediTwin AI Streamlit UI..."
    streamlit run frontend/app.py --server.port 8501
    ;;
  backend)
    echo "🔌 Starting FastAPI backend..."
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    ;;
  train)
    echo "🧠 Training tabular model..."
    python scripts/train_models.py
    ;;
  train-all)
    echo "🧠 Training all models..."
    python scripts/train_models.py --imaging
    ;;
  test)
    echo "🧪 Running test suite..."
    pytest tests/ -v
    ;;
  *)
    echo "Usage: ./run.sh [streamlit|backend|train|train-all|test]"
    ;;
esac
