import os
import types
import sys
sys.path.insert(0, r'e:autonomous_multi_research_agent\backend')
from app.services.llm_provider import summarize_with_provider
plan = types.SimpleNamespace(objective='test')
docs = [types.SimpleNamespace(title='t1', url='u1', abstract='a1', content='c1', source='arxiv')]
print('Calling summarize_with_provider...')
res = summarize_with_provider(plan, docs, model_name='gpt-3.5-turbo')
print('Result head:', res[:200])
print('Logs dir exists:', os.path.isdir(r'e:autonomous_multi_research_agent\backend\logs'))
print('Logs listing:', os.listdir(r'e:autonomous_multi_research_agent\backend\logs'))
