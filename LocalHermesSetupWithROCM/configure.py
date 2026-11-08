#!/usr/bin/env python3
"""Merge local-model configuration into Hermes; preserve unrelated settings."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import urllib.request
import urllib.error

try:
    import yaml
except ImportError:
    raise SystemExit('PyYAML is required. Install python-yaml on Arch before configuring.')


def read_json(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return data


def save(path, text):
    """Back up changed files, then replace atomically; unchanged files stay alone."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f'Refusing to replace symlink {path}; edit its target manually')
    if path.exists() and path.read_text() == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    if path.exists():
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        shutil.copy2(path, path.with_name(path.name + '.backup-' + stamp))
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            output.write(text)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f'Updated {path}')


def configure(home, models, default, context=65536, hermes_home=None, check_only=False):
    if not models or len(set(models)) != len(models) or default not in models:
        raise ValueError('Models must be unique and include the default')
    if context <= 0:
        raise ValueError('Context length must be positive')
    hermes_home = hermes_home if hermes_home is not None else home / '.hermes'
    ollama_path = home / '.ollama/config.json'
    hermes_path = hermes_home / 'config.yaml'
    # Parse and validate both files before making any writes.
    ollama = read_json(ollama_path)
    hermes = yaml.safe_load(hermes_path.read_text()) if hermes_path.exists() else {}
    if hermes is None:
        hermes = {}
    if not isinstance(hermes, dict):
        raise ValueError(f'{hermes_path} must contain a YAML mapping')
    model = hermes.get('model', {})
    # Hermes also accepts a bare model name in older configurations.
    if isinstance(model, str):
        model = {'default': model}
    if not isinstance(model, dict):
        raise ValueError(f'{hermes_path}: model must be a mapping or string')
    # YAML aliases can share this mapping with auxiliary/provider settings.
    model = dict(model)
    model.update({'default': default, 'provider': 'custom',
                  'base_url': 'http://127.0.0.1:11434/v1',
                  'api_key': 'ollama', 'api_mode': 'chat_completions',
                  'context_length': context})
    hermes['model'] = model
    display = hermes.get('display', {})
    if not isinstance(display, dict):
        raise ValueError(f'{hermes_path}: display must be a mapping')
    hermes['display'] = {**display, 'skin': 'white'}
    white_skin = (Path(__file__).resolve().parent / 'skins/white.yaml').read_text()
    integrations = ollama.setdefault('integrations', {})
    if not isinstance(integrations, dict):
        raise ValueError(f'{ollama_path}: integrations must be an object')
    if not isinstance(integrations.get('hermes', {}), dict):
        raise ValueError(f'{ollama_path}: integrations.hermes must be an object')
    ordered = [default] + [name for name in models if name != default]
    integrations.setdefault('hermes', {})['models'] = ordered
    providers = hermes.get('custom_providers', [])
    if not isinstance(providers, list) or any(not isinstance(p, dict) for p in providers):
        raise ValueError(f'{hermes_path}: custom_providers must be a list of mappings')
    providers = [dict(p) for p in providers]
    local = next((p for p in providers if p.get('name') == 'Local Ollama'
                  and p.get('base_url', '').rstrip('/') == model['base_url']), None)
    if local is None:
        local = {'name': 'Local Ollama'}
        providers.append(local)
    local.update({'base_url': model['base_url'], 'api_key': 'ollama',
                  'api_mode': 'chat_completions', 'model': default,
                  'models': {name: {'context_length': context} for name in ordered}})
    hermes['custom_providers'] = providers
    pending = {hermes_home / 'skins/white.yaml': white_skin,
               ollama_path: json.dumps(ollama, indent=2) + '\n',
               hermes_path: yaml.safe_dump(hermes, sort_keys=False, allow_unicode=True)}

    # Avoid global API environment exports; configure each agent explicitly.
    path_script = '''# Managed by LocalModelSetupScript. Bash login/interactive PATH.
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac
'''
    pending[home / '.config/local-model-setup/path.sh'] = path_script
    source_line = '[ ! -f "$HOME/.config/local-model-setup/path.sh" ] || . "$HOME/.config/local-model-setup/path.sh"'
    for name in ['.bashrc', '.bash_profile']:
        path = home / name
        existing = path.read_text() if path.exists() else ''
        # Preserve Bash's login-file precedence when introducing .bash_profile.
        if name == '.bash_profile' and not path.exists():
            existing = ('if [ -f "$HOME/.bash_login" ]; then\n'
                        '  . "$HOME/.bash_login"\n'
                        'elif [ -f "$HOME/.profile" ]; then\n'
                        '  . "$HOME/.profile"\n'
                        'fi\n')
            existing += '[ ! -f "$HOME/.bashrc" ] || . "$HOME/.bashrc"\n'
        if source_line not in existing.splitlines():
            pending[path] = existing.rstrip('\n') + '\n\n# Local model agents\n' + source_line + '\n'
    pending[home / '.config/fish/conf.d/local-model-setup.fish'] = (
         '# Managed by LocalModelSetupScript.\n'
         'fish_add_path --path "$HOME/.local/bin"\n')

    # Validate the complete output set before touching any files.
    for path in pending:
        if path.is_symlink():
            raise ValueError(f'Refusing to replace symlink {path}; edit its target manually')
        if path.exists() and not path.is_file():
            raise ValueError(f'{path} must be a regular file')
        for parent in path.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f'{parent} must be a directory')
    if not check_only:
        for path, text in pending.items():
            save(path, text)


def installed_models():
    """Read the local server catalog; fall back to Arch's on-disk catalog."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open('http://127.0.0.1:11434/api/tags', timeout=5) as response:
            return [m['name'] for m in json.load(response).get('models', [])
                    if isinstance(m.get('name'), str) and not m.get('remote_host')]
    except (OSError, ValueError, urllib.error.URLError):
        roots = [Path(os.environ.get('OLLAMA_MODELS', '/var/lib/ollama')),
                 Path.home() / '.ollama/models']
        names = []
        for root in roots:
            manifests = root / 'manifests'
            for path in sorted(manifests.glob('*/*/*/*')):
                if not path.is_file():
                    continue
                registry, namespace, model, tag = path.relative_to(manifests).parts
                if tag == 'cloud':
                    continue
                name = ('' if namespace == 'library' else namespace + '/') + model + ':' + tag
                if registry != 'registry.ollama.ai':
                    name = registry + '/' + name
                names.append(name)
        return list(dict.fromkeys(names))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--default', required=True)
    parser.add_argument('--context', type=int, default=65536)
    parser.add_argument('--hermes-home', type=Path)
    parser.add_argument('--discover-installed', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.discover_installed:
        args.models = list(dict.fromkeys(args.models + installed_models()))
    configure(args.home, args.models, args.default, args.context, args.hermes_home, args.check_only)


if __name__ == '__main__':
    main()
