import importlib, traceback, sys, os

def probe(name):
    try:
        m = importlib.import_module(name)
        print(f"MODULE:{name}")
        print("file:", getattr(m, '__file__', None))
        print("version:", getattr(m, '__version__', None))
        attrs = sorted([a for a in dir(m) if not a.startswith('_')])
        print('attrs_count:', len(attrs))
        print('has_generate_text:', hasattr(m, 'generate_text'))
        print('has_Text_generate:', hasattr(m, 'Text') and hasattr(getattr(m,'Text'), 'generate'))
        print('has_generate:', hasattr(m, 'generate'))
        print('has_chat_create:', hasattr(m, 'chat') and hasattr(getattr(m,'chat'), 'create'))
        print('has_chats_create:', hasattr(m, 'chats') and hasattr(getattr(m,'chats'), 'create'))
        print('has_models_generate:', hasattr(m, 'models') and hasattr(getattr(m,'models'), 'generate'))
        print('has_client_generate:', hasattr(m, 'client') and hasattr(getattr(m,'client'), 'generate'))
        # print first 200 attrs
        print('attrs_sample:', attrs[:200])
    except Exception as e:
        print(f"IMPORT_ERROR:{name}:{type(e).__name__}:{e}")
        traceback.print_exc()

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__) or '.')
    for n in ('google.genai','google.generativeai'):
        probe(n)
    # also try top-level 'genai' if present
    probe('genai')
    sys.exit(0)
