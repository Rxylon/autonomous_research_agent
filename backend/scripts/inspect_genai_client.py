import importlib, traceback, sys, os

try:
    import google.genai as genai
    print('genai version', getattr(genai, '__version__', None))
    client_mod = genai.client
    print('client_mod:', client_mod)
    ClientClass = getattr(client_mod, 'Client', None) or getattr(client_mod, 'GeminiNextGenAPIClient', None) or getattr(client_mod, 'GeminiNextGenAPIClientAdapter', None)
    print('ClientClass:', ClientClass)
    if ClientClass:
        print('Client attrs sample:', sorted([a for a in dir(ClientClass) if not a.startswith('_')])[:200])
    # inspect chats and models
    print('chats attrs sample:', sorted([a for a in dir(genai.chats) if not a.startswith('_')])[:200])
    print('models attrs sample:', sorted([a for a in dir(genai.models) if not a.startswith('_')])[:200])
except Exception as e:
    print('ERROR', type(e).__name__, e)
    traceback.print_exc()
