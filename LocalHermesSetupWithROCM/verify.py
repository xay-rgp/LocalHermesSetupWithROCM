#!/usr/bin/env python3
"""Smoke-test local inference and confirm ROCm appears in the current service log."""
import argparse
import json
import shlex
import subprocess
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--context', type=int, required=True)
    parser.add_argument('--kv-cache', choices=['f16', 'q8_0', 'q4_0'])
    parser.add_argument('--flash-attention', choices=['0', '1'])
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = 'http://127.0.0.1:11434'

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=900) as response:
            result = json.load(response)
        if result.get('error'):
            raise RuntimeError(result['error'])
        return result

    # Check the effective service environment, including later drop-ins.
    environment = subprocess.check_output(
        ['systemctl', 'show', 'ollama.service', '--property=Environment', '--value'],
        text=True)
    effective = dict(item.split('=', 1) for item in shlex.split(environment) if '=' in item)
    expected = {
        'OLLAMA_CONTEXT_LENGTH': str(args.context),
        'OLLAMA_HOST': '127.0.0.1:11434',
        'OLLAMA_VULKAN': '0',
        'OLLAMA_NUM_PARALLEL': '1',
        'OLLAMA_MAX_LOADED_MODELS': '1',
    }
    if args.kv_cache is not None:
        expected['OLLAMA_KV_CACHE_TYPE'] = args.kv_cache
    if args.flash_attention is not None:
        expected['OLLAMA_FLASH_ATTENTION'] = args.flash_attention
    for key, value in expected.items():
        if effective.get(key) != value:
            raise SystemExit(f'{key}={effective.get(key)!r}, expected {value!r}. '
                             'Check later systemd drop-ins before updating agents.')

    info = request('/api/show', {'model': args.model})
    if 'tools' not in info.get('capabilities', []):
        raise SystemExit('The default model does not advertise tool calling; choose an Ollama model with tools support.')

    result = request('/api/generate', {
        'model': args.model, 'prompt': 'Say hello briefly.', 'stream': False,
        'options': {'num_predict': 16, 'temperature': 0},
    })
    if not result.get('done') or not (result.get('response') or result.get('thinking')):
        raise SystemExit('Ollama did not return a completed, nonempty generation.')
    # Canonical IDs can append :latest, so compare the server's response ID.
    model_name = result.get('model', args.model)
    canonical = lambda name: name if ':' in name.rsplit('/', 1)[-1] else name + ':latest'
    loaded = [item for item in request('/api/ps').get('models', [])
              if canonical(item.get('name', '')) == canonical(model_name)]
    if not loaded or loaded[0].get('size_vram', 0) <= 0:
        raise SystemExit('Model generated, but GPU allocation was not detected. Check journalctl -u ollama.')
    actual_context = loaded[0].get('context_length')
    if actual_context != args.context:
        raise SystemExit(f'Ollama loaded context={actual_context}, expected {args.context}. '
                         'Check service overrides and model num_ctx parameters.')
    print(f'Loaded context verified: {actual_context} tokens')
    # Restrict logs to the current systemd invocation; old ROCm runs are not evidence.
    invocation = subprocess.check_output(
        ['systemctl', 'show', 'ollama.service', '--property=InvocationID', '--value'], text=True).strip()
    if not invocation:
        raise SystemExit('Cannot identify the current Ollama service invocation.')
    logs = subprocess.check_output(
        ['sudo', 'journalctl', f'_SYSTEMD_INVOCATION_ID={invocation}', '--no-pager', '-o', 'cat'],
        text=True, errors='replace')
    if 'library=ROCm' not in logs or 'ROCm0' not in logs:
        raise SystemExit('GPU allocation exists, but ROCm was not verified in current logs. Inspect journalctl -u ollama.')
    print(f"ROCm verified: {args.model}, GPU allocation {loaded[0]['size_vram'] / 2**30:.2f} GiB")
    if result.get('eval_duration'):
        print(f"Short smoke test: {result.get('eval_count', 0) * 1e9 / result['eval_duration']:.1f} tokens/s")
    print('This checks inference, not agent tool-call quality or a full performance benchmark.')


if __name__ == '__main__':
    main()
