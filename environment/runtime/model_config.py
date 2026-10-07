"""Allowlisted Codex model/provider configuration; no tools or sandbox overrides."""
import copy
import json
import os
from pathlib import Path
import shutil
import tomllib

PROVIDER_KEYS = {'name','base_url','wire_api','env_key','experimental_bearer_token','requires_openai_auth','supports_websockets','request_max_retries','stream_max_retries','stream_idle_timeout_ms','websocket_connect_timeout_ms','http_headers','env_http_headers','query_params'}

def filtered_config(config, model=None, environ=None):
    env = os.environ if environ is None else environ
    result = {k:config[k] for k in ('model','model_reasoning_effort','model_provider') if isinstance(config.get(k),str)}
    if model is not None:result['model']=model
    provider = result.get('model_provider','openai')
    providers = config.get('model_providers',{})
    if provider in providers:
        original = providers[provider]
        extra = set(original)-PROVIDER_KEYS
        if extra:raise ValueError('Unsupported provider fields: '+', '.join(sorted(extra)))
        settings = copy.deepcopy(original)
        if settings.get('wire_api','responses') != 'responses':
            raise ValueError('This pinned Codex integration requires the Responses API, not chat/completions')
        key = settings.pop('env_key',None)
        if key:
            if not env.get(key):raise ValueError('Missing model credential environment variable: '+key)
            if 'experimental_bearer_token' in settings:raise ValueError('Choose env_key or bearer token, not both')
            settings['experimental_bearer_token']=env[key]
        headers=settings.setdefault('http_headers',{})
        for header,variable in settings.pop('env_http_headers',{}).items():
            if not env.get(variable):raise ValueError('Missing provider header environment variable: '+variable)
            headers[header]=env[variable]
        if not headers:settings.pop('http_headers')
        result['model_providers']={provider:settings}
    elif provider!='openai':
        raise ValueError('Define the custom provider under model_providers: '+provider)
    return result

def toml_text(config):
    # Only the allowlisted scalar and provider dictionary structures above.
    lines=[]
    def table(obj,path):
        if path:lines.append('['+'.'.join(json.dumps(x) for x in path)+']')
        for key,value in obj.items():
            if not isinstance(value,dict):
                if not isinstance(value,(str,int,float,bool)):raise ValueError('Unsupported TOML value: '+key)
                lines.append(json.dumps(key)+' = '+json.dumps(value))
        for key,value in obj.items():
            if isinstance(value,dict):table(value,[*path,key])
    table(config,[])
    return '\n'.join(lines)+'\n'

def prepare_model_home(source,destination,model=None):
    source,destination=Path(source),Path(destination)
    cfg=source/'config.toml'
    config=filtered_config(tomllib.loads(cfg.read_text()) if cfg.exists() else {},model=model)
    provider=config.get('model_provider','openai')
    info=config.get('model_providers',{}).get(provider,{})
    needs_login=info.get('requires_openai_auth',provider=='openai')
    destination.mkdir(parents=True,exist_ok=True);destination.chmod(0o700)
    if (source/'auth.json').exists():
        shutil.copyfile(source/'auth.json',destination/'auth.json');(destination/'auth.json').chmod(0o600)
    elif needs_login:raise FileNotFoundError('Codex login required: '+str(source/'auth.json'))
    target=destination/'config.toml';target.write_text(toml_text(config));target.chmod(0o600)
    return {'model':config.get('model'),'provider':provider,'reasoning_effort':config.get('model_reasoning_effort')}
