"""Azure OpenAI chat helper. Credentials: AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY in the environment, or the file named by
FEDAI_AZURE_ENV (default ~/.ssh/azure_fedai.env). They are never written anywhere else."""
import json, os, time, urllib.request
_f = os.path.expanduser(os.environ.get('FEDAI_AZURE_ENV', '~/.ssh/azure_fedai.env'))
ENV = dict(l.strip().split('=', 1) for l in open(_f) if '=' in l) if os.path.exists(_f) else {}
ENV.update({k: v for k, v in os.environ.items() if k.startswith(('AZURE_OPENAI_', 'FEDAI_RAG_'))})
EP = ENV['AZURE_OPENAI_ENDPOINT'].rstrip('/'); KEY = ENV['AZURE_OPENAI_API_KEY']
MODEL = os.environ.get('LLM_MODEL', ENV.get('FEDAI_RAG_CHAT_MODEL', 'gpt-4.1-mini')); VER = ENV.get('AZURE_OPENAI_API_VERSION', '2024-10-21')
USAGE = {'in': 0, 'out': 0}
REASONING = ('o4-mini', 'o3', 'gpt-5-mini', 'gpt-5', 'gpt-5.6-luna')
def chat(system, user, temperature=0.0, max_tokens=600, model=None):
    model = model or MODEL
    msg = {'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], 'response_format': {'type': 'json_object'}}
    if model.startswith(REASONING): msg['max_completion_tokens'] = max(8000, max_tokens * 6)      # reasoning models: no temperature
    else: msg.update(temperature=temperature, max_tokens=max_tokens)
    body = json.dumps(msg).encode()
    url = f'{EP}/openai/deployments/{model}/chat/completions?api-version=2025-01-01-preview'
    for a in range(int(os.environ.get('LLM_RETRIES', 6))):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json', 'api-key': KEY}), timeout=120))
            USAGE['in'] += r['usage']['prompt_tokens']; USAGE['out'] += r['usage']['completion_tokens']
            USAGE.setdefault(model, [0, 0]); USAGE[model][0] += r['usage']['prompt_tokens']; USAGE[model][1] += r['usage']['completion_tokens']
            return json.loads(r['choices'][0]['message']['content'])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                ra = e.headers.get('retry-after') if e.headers else None
                time.sleep(min(90, float(ra) + 1 if ra and ra.replace('.', '').isdigit() else 5 * (a + 1))); continue
            raise RuntimeError(f'HTTP {e.code}: {e.read()[:300]}')
    raise RuntimeError('Azure retries exhausted')
