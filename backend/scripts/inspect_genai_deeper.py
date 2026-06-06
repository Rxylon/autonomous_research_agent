import importlib, traceback, sys, os

def probe(name):
    try:
        m = importlib.import_module(name)
        print(f"MODULE:{name} file={getattr(m,'__file__',None)} version={getattr(m,'__version__',None)}")
        for sub in ('models','chats','client','chat','Text'):
            try:
                submod = getattr(m, sub, None)
                if submod is None:
                    print(f" {sub}: NOT_PRESENT")
                else:
                    print(f" {sub}: present type={type(submod)} attrs_sample={sorted([a for a in dir(submod) if not a.startswith('_')])[:40]}")
            except Exception as e:
                print(f" {sub}: ERROR {type(e).__name__} {e}")
    except Exception as e:
        print(f"IMPORT_ERROR:{name}:{type(e).__name__}:{e}")
        traceback.print_exc()

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__) or '.')
    for n in ('google.genai','google.generativeai'):
        probe(n)
    sys.exit(0)
