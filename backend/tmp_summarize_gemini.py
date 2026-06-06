import os, types, sys
sys.path.insert(0, r'e:utonomous_multi_research_agent\backend')
# set env to force gemini provider
os.environ['DEFAULT_LLM_PROVIDER']='gemini'
os.environ['GEMINI_API_KEY']='DUMMY_KEY'
from app.services.llm_provider import summarize_with_provider
plan = types.SimpleNamespace(objective='test gemini')
docs = [types.SimpleNamespace(title='t1', url='u1', abstract='a1', content='c1', source='arxiv')]
print('Calling summarize_with_provider (gemini)...')
res = summarize_with_provider(plan, docs, model_name='gemini-1.0')
print('Result head:', res[:200])
print('Logs listing:', os.listdir(r'e:autonomous_multi_research_agent\backend\logs'))
