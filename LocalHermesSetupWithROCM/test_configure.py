"""Offline tests: configuration preservation, reruns and model changes."""
import contextlib
import io
import json
import subprocess
from pathlib import Path
import tempfile
import unittest

import yaml

from configure import configure


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.models = ['qwen3-coder:30b', 'huihui_ai/Qwen3.8-abliterated:27b']

    def write(self, name, data):
        path = self.home / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def read(self, name):
        return yaml.safe_load((self.home / name).read_text())

    def run_configuration(self, models=None):
        models = self.models if models is None else models
        with contextlib.redirect_stdout(io.StringIO()):
            configure(self.home, models, models[0])

    def test_pi_configuration_is_not_created(self):
        self.run_configuration()
        self.assertFalse((self.home / '.pi').exists())

    def test_existing_pi_configuration_is_untouched(self):
        path = self.home / '.pi/agent/settings.json'
        path.parent.mkdir(parents=True)
        path.write_text('{not valid json')
        before = (path.read_bytes(), path.stat().st_mtime_ns)
        existing = {'models': ['previous:model'], 'custom': True}
        self.write('.ollama/config.json', {'integrations': {'pi': existing}})
        self.run_configuration()
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        self.assertEqual(list(path.parent.iterdir()), [path])
        self.assertEqual(self.read('.ollama/config.json')['integrations']['pi'], existing)

    def test_claude_configuration_is_not_created(self):
        self.run_configuration()
        self.assertFalse((self.home / '.claude').exists())
        self.assertEqual(self.read('.ollama/config.json')['integrations'], {
            'hermes': {'models': self.models},
        })

    def test_existing_claude_configuration_is_untouched(self):
        path = self.home / '.claude/settings.json'
        path.parent.mkdir()
        # Even invalid Claude JSON must not prevent configuring Hermes.
        path.write_text('{not valid json')
        before = (path.read_bytes(), path.stat().st_mtime_ns)
        existing = {'models': ['previous:model'], 'custom': True}
        self.write('.ollama/config.json', {'integrations': {'claude': existing}})
        self.run_configuration()
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        self.assertEqual(list(path.parent.iterdir()), [path])
        self.assertEqual(self.read('.ollama/config.json')['integrations']['claude'], existing)
        self.assertEqual(self.read('.ollama/config.json')['integrations']['hermes']['models'], self.models)

    def test_hermes_configuration_preserves_settings_and_backs_up(self):
        original = {'model': {'default': 'old', 'provider': 'openrouter',
                              'api_key': 'old-secret', 'api_mode': 'anthropic_messages', 'custom': True},
                    'terminal': {'backend': 'docker'},
                    'providers': {'other': {'api_key': 'keep-me'}}}
        path = self.home / '.hermes/config.yaml'
        path.parent.mkdir()
        path.write_text(yaml.safe_dump(original))
        self.run_configuration()
        hermes = self.read('.hermes/config.yaml')
        self.assertEqual(hermes['terminal'], original['terminal'])
        self.assertEqual(hermes['providers'], original['providers'])
        self.assertEqual(hermes['model'], {
            'default': self.models[0], 'provider': 'custom', 'custom': True,
            'base_url': 'http://127.0.0.1:11434/v1', 'api_key': 'ollama',
            'api_mode': 'chat_completions', 'context_length': 65536})
        backups = list(path.parent.glob('config.yaml.backup-*'))
        self.assertEqual(yaml.safe_load(backups[0].read_text()), original)
        self.run_configuration(['example/new:latest'])
        self.assertEqual(self.read('.hermes/config.yaml')['model']['default'],
                         'example/new:latest')

    def test_yaml_alias_does_not_change_auxiliary_provider(self):
        path = self.home / '.hermes/config.yaml'
        path.parent.mkdir()
        path.write_text('model: &shared\n  default: old-model\n  provider: openrouter\n'
                        'auxiliary:\n  compression: *shared\n')
        original = yaml.safe_load(path.read_text())
        self.run_configuration()
        result = self.read('.hermes/config.yaml')
        self.assertEqual(result['auxiliary'], original['auxiliary'])
        self.assertEqual(result['model']['provider'], 'custom')
        self.assertEqual(result['model']['context_length'], 65536)
        self.run_configuration()
        self.assertEqual(self.read('.hermes/config.yaml'), result)

    def test_stale_context_is_replaced_without_changing_compression(self):
        compression = {'enabled': True, 'threshold': 0.5, 'protect_last_n': 20}
        self.write('.hermes/config.yaml', {
            'model': {'context_length': 262144}, 'compression': compression})
        for context in (65536, 32768):
            with self.subTest(context=context), contextlib.redirect_stdout(io.StringIO()):
                configure(self.home, self.models, self.models[0], context)
                config = self.read('.hermes/config.yaml')
                self.assertEqual(config['model']['context_length'], context)
                self.assertEqual(config['compression'], compression)

    def test_bad_hermes_yaml_fails_before_any_writes(self):
        path = self.home / '.hermes/config.yaml'
        path.parent.mkdir()
        path.write_text('model: [broken')
        with self.assertRaises(yaml.YAMLError):
            self.run_configuration()
        self.assertFalse((self.home / '.ollama').exists())
        self.assertFalse((self.home / '.bashrc').exists())
        self.assertFalse(list(self.home.rglob('*.backup-*')))

    def test_cline_configuration_is_untouched(self):
        path = self.home / '.cline/data/settings/providers.json'
        path.parent.mkdir(parents=True)
        path.write_text('{invalid json')
        before = (path.read_bytes(), path.stat().st_mtime_ns)
        self.write('.ollama/config.json', {'integrations': {'cline': {'custom': True}}})
        self.run_configuration()
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        self.assertEqual(self.read('.ollama/config.json')['integrations']['cline'],
                         {'custom': True})

    def test_custom_home_context_and_legacy_model(self):
        custom = self.home / 'custom-hermes'
        custom.mkdir()
        (custom / 'config.yaml').write_text('model: old-model\n')
        with contextlib.redirect_stdout(io.StringIO()):
            configure(self.home, self.models, self.models[0], 131072, custom)
        model = self.read('custom-hermes/config.yaml')['model']
        self.assertEqual(model['context_length'], 131072)
        self.assertEqual(model['default'], self.models[0])
        self.assertFalse((self.home / '.hermes').exists())

    def test_preflight_makes_no_writes(self):
        configure(self.home, self.models, self.models[0], check_only=True)
        self.assertFalse(list(self.home.iterdir()))

    def test_late_symlink_fails_without_partial_writes(self):
        target = self.home / 'original'
        target.write_text('# original\n')
        (self.home / '.bashrc').symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.run_configuration()
        self.assertFalse((self.home / '.ollama').exists())
        self.assertFalse((self.home / '.hermes').exists())
        self.assertEqual(target.read_text(), '# original\n')

    def test_invalid_nested_structures(self):
        for bad in ({'integrations': []}, {'integrations': {'hermes': None}}):
            self.write('.ollama/config.json', bad)
            with self.assertRaisesRegex(ValueError, 'must be an object'):
                self.run_configuration()
            self.assertFalse((self.home / '.hermes').exists())
        (self.home / '.ollama/config.json').unlink()
        for bad in ([], {'model': []}):
            self.write('.hermes/config.yaml', bad)
            with self.assertRaises(ValueError):
                self.run_configuration()
            self.assertFalse((self.home / '.ollama/config.json').exists())

    def test_file_modes_are_preserved(self):
        self.run_configuration()
        path = self.home / '.hermes/config.yaml'
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        path.chmod(0o640)
        self.run_configuration(['changed:model'])
        self.assertEqual(path.stat().st_mode & 0o777, 0o640)
        self.assertEqual(next(path.parent.glob('*.backup-*')).stat().st_mode & 0o777, 0o640)

    def test_second_run_changes_nothing(self):
        self.run_configuration()
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.home.rglob('*') if p.is_file()}
        self.run_configuration()
        after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.home.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_invalid_model_lists_are_rejected(self):
        for models, default in [(['a', 'a'], 'a'), ([], 'a'), (['a'], 'b')]:
            with self.assertRaises(ValueError):
                configure(self.home, models, default)
        self.assertFalse(list(self.home.iterdir()))

    def test_default_model_is_first_in_launcher(self):
        with contextlib.redirect_stdout(io.StringIO()):
            configure(self.home, self.models, self.models[1])
        self.assertEqual(self.read('.ollama/config.json')['integrations']['hermes']['models'],
                         list(reversed(self.models)))
        self.assertEqual(self.read('.hermes/config.yaml')['model']['default'], self.models[1])

    def test_new_login_profile_preserves_bash_login_precedence(self):
        (self.home / '.bash_login').write_text('export LOGIN_MARKER=bash_login\n')
        (self.home / '.profile').write_text('export LOGIN_MARKER=profile\n')
        self.run_configuration()
        result = subprocess.run(
            ['bash', '--noprofile', '--norc', '-c',
             '. "$HOME/.bash_profile"; printf "%s" "$LOGIN_MARKER"'],
            env={'HOME': str(self.home), 'PATH': '/usr/bin:/bin'},
            capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, 'bash_login')

    def test_new_login_profile_preserves_profile_loading(self):
        (self.home / '.profile').write_text('# existing profile\n')
        self.run_configuration()
        profile = (self.home / '.bash_profile').read_text()
        self.assertIn('. "$HOME/.profile"', profile)
        self.assertIn('. "$HOME/.bashrc"', profile)
        self.assertIn('fish_add_path', (self.home / '.config/fish/conf.d/local-model-setup.fish').read_text())



if __name__ == '__main__':
    unittest.main()
