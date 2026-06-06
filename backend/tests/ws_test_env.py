from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# ensure the backend package is importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

load_dotenv(dotenv_path=repo_root.joinpath('.env'))

from app.core.config import settings
from app.main import app

print('LLM provider:', settings.default_llm_provider)
print('OPENAI key set:', bool(settings.openai_api_key))
print('GEMINI key set:', bool(settings.gemini_api_key))

client = TestClient(app)
try:
    with client.websocket_connect('/ws/research') as ws:
        ws.send_json({'query': 'Find recent advances in multimodal deepfake detection and summarize major approaches.'})
        while True:
            msg = ws.receive_json()
            print(msg)
            if msg.get('type') == 'result':
                break
except Exception as e:
    print('WebSocket test failed:', e)
