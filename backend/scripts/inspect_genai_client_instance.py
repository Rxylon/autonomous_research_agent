import traceback
try:
    import google.genai as genai
    Client = genai.client.Client
    c = Client()
    print('Client instance created:', type(c))
    print('dir(client) sample:', sorted([a for a in dir(c) if not a.startswith('_')])[:200])
    chats = getattr(c, 'chats', None)
    models = getattr(c, 'models', None)
    print('client.chats:', type(chats))
    if chats is not None:
        print('client.chats attrs sample:', sorted([a for a in dir(chats) if not a.startswith('_')])[:200])
    print('client.models:', type(models))
    if models is not None:
        print('client.models attrs sample:', sorted([a for a in dir(models) if not a.startswith('_')])[:200])
except Exception as e:
    print('ERROR', type(e).__name__, e)
    traceback.print_exc()
