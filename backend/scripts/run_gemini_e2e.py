import os, sys, traceback
# load .env manually to ensure env vars are set for runtime
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.llm_provider import summarize_with_provider

class Doc:
    def __init__(self, title, content, source='local'):
        self.title = title
        self.content = content
        self.source = source
        self.url = ''
        self.abstract = ''


docs = [Doc('Test paper 1', 'This is a snippet about transformer models and alignment.'), Doc('Test paper 2', 'This discusses multimodal fusion and robustness.')]

print('Using GEMINI_API_KEY set?', bool(os.environ.get('GEMINI_API_KEY')))
try:
    out = summarize_with_provider(plan=None, documents=docs)
    print('=== PROVIDER OUTPUT START ===')
    print(out)
    print('=== PROVIDER OUTPUT END ===')
except Exception:
    print('Exception during summarize_with_provider:')
    traceback.print_exc()

# dump recent logs
logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
for fname in ('llm_provider_trace.log', 'llm_provider_errors.log'):
    path = os.path.join(logs_dir, fname)
    print('\n---', fname, '---')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fh:
            lines = fh.read().splitlines()[-40:]
            print('\n'.join(lines))
    else:
        print('missing')
