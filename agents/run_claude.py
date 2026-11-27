"""Run the unchanged CC wrapper, optionally with a private local protocol converter."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import urllib.request

def provider_proxy_config(model):
    """Declare a Chat Completions endpoint, never infer Responses support.

    LiteLLM 1.105.1 otherwise sends Anthropic Messages for the ``openai``
    provider to /responses. This profile explicitly accepts a chat base URL;
    preserve translated reasoning/tool parameters and let that provider validate
    them instead of silently dropping parameters or declaring invented abilities.
    """
    return {
        'model_list': [{'model_name': model, 'litellm_params': {
            'model': 'openai/' + model,
            'api_base': 'os.environ/WORLD_HARBOR_CHAT_BASE_URL',
            'api_key': 'os.environ/WORLD_MODEL_API_KEY'}}],
        'litellm_settings': {
            'use_chat_completions_url_for_anthropic_messages': True,
            'drop_params': False, 'num_retries': 0, 'telemetry': False},
        'general_settings': {'master_key': 'os.environ/WORLD_PROVIDER_PROXY_KEY'},
    }


def configure_chat_provider():
    """Make the pinned converter honour its chat opt-out for thinking too.

    1.105.1's Messages adapter has a second Responses-routing helper which
    ignores the global opt-out, changes the model name, and wraps the plain
    reasoning_effort in a Responses-only dict. Guard only that routing helper;
    leave the actual thinking -> reasoning_effort and tool/image mappers intact.
    This is an external compatibility shim, not a harness/model/source patch.
    """
    from importlib.metadata import version
    if version('litellm') != '1.105.1':
        raise RuntimeError('Revalidate the Chat provider shim before changing LiteLLM 1.105.1')
    import litellm
    from litellm.llms.anthropic.pass_through.adapters.handler import (
        LiteLLMMessagesToCompletionTransformationHandler as Handler,
    )
    original = Handler._route_openai_thinking_to_responses_api_if_needed
    if getattr(original, '_robotworld_chat_guard', False):
        return

    def route(completion_kwargs, *, thinking):
        if litellm.use_chat_completions_url_for_anthropic_messages:
            return
        return original(completion_kwargs, thinking=thinking)

    route._robotworld_chat_guard = True
    Handler._route_openai_thinking_to_responses_api_if_needed = staticmethod(route)


def run_provider_server():
    configure_chat_provider()
    from litellm import run_server
    run_server()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--instruction-file',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    env=os.environ.copy()
    proxy=None;log=None
    try:
        base=env.get('WORLD_HARBOR_CHAT_BASE_URL')
        if base:
            # Keep LiteLLM's dependencies separate from the MCP 1.x agent environment.
            cfg=output.parent/'provider-proxy.json'
            cfg.write_text(json.dumps(provider_proxy_config(args.model), indent=2))
            env['WORLD_PROVIDER_PROXY_KEY']='sk-'+secrets.token_hex(24)
            log=(output.parent/'provider-proxy.log').open('w')
            proxy=subprocess.Popen(['/opt/robotworld-provider/bin/python','-c',
                'from agents.run_claude import run_provider_server; run_provider_server()',
                '--config',str(cfg),
                '--host','127.0.0.1','--port','18881'],env=env,stdout=log,stderr=subprocess.STDOUT)
            for _ in range(120):
                if proxy.poll() is not None:raise RuntimeError('Provider converter exited; inspect provider-proxy.log')
                try:
                    req=urllib.request.Request('http://127.0.0.1:18881/health/liveliness',
                        headers={'Authorization':'Bearer '+env['WORLD_PROVIDER_PROXY_KEY']})
                    with urllib.request.urlopen(req,timeout=1) as response:
                        if response.status==200:break
                except (OSError,ValueError):time.sleep(1)
            else:raise TimeoutError('Provider converter readiness timeout')
            env['ANTHROPIC_BASE_URL']='http://127.0.0.1:18881'
            env['ANTHROPIC_API_KEY']=env['WORLD_PROVIDER_PROXY_KEY']
        result=subprocess.run([sys.executable,'-m','harnesses.claude_code.run',
            '--model',args.model,'--output',str(output),'--wall-timeout','43200','--instruction-file',str(args.instruction_file)],env=env)
        if result.returncode:raise RuntimeError('Claude Code wrapper exited nonzero')
    finally:
        if proxy:
            proxy.terminate()
            try:proxy.wait(timeout=10)
            except subprocess.TimeoutExpired:proxy.kill();proxy.wait()
        if log:log.close()
if __name__=='__main__':main()
