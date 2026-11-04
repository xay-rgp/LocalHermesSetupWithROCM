"""Exercise shell modes with temporary homes and simulated system commands."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import yaml


class SetupTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        for name in ('setup.sh', 'config.sh', 'configure.py'):
            shutil.copy2(Path(__file__).parent / name, self.repo / name)
        (self.repo / 'verify.py').write_text(
            'import os, sys\nfrom pathlib import Path\n'
            'with open(os.environ["AUDIT_LOG"], "a") as f: f.write("verify\\n")\n'
            'sys.exit(int(os.environ.get("VERIFY_FAIL", "0")))\n')
        self.home = self.root / 'home'
        self.home.mkdir()
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.log = self.root / 'calls'
        self.env = dict(os.environ, HOME=str(self.home),
                        HERMES_HOME=str(self.home / '.hermes'),
                        PATH=f'{self.bin}:/usr/bin:/bin', AUDIT_LOG=str(self.log))
        self.stub('sudo', '''printf 'sudo %s\\n' "$*" >> "$AUDIT_LOG"
if [[ "$1" == test ]]; then exit 1; fi
if [[ "$1" == pacman && "${PACMAN_FAIL:-0}" == 1 ]]; then exit 1; fi
''')
        self.stub('ollama', 'echo "ollama $*" >> "$AUDIT_LOG"\n')
        self.stub('hermes', 'echo "hermes $*" >> "$AUDIT_LOG"\n')
        self.stub('curl', '''echo "curl $*" >> "$AUDIT_LOG"
if [[ "$*" == *api/version* ]]; then exit 0; fi
if [[ "${DOWNLOAD_FAIL:-0}" == 1 ]]; then exit 22; fi
while (($#)); do
  if [[ "$1" == --output ]]; then
    printf '%s\\n' '#!/bin/bash' 'echo "installer $*" >> "$AUDIT_LOG"' 'exit "${INSTALL_FAIL:-0}"' > "$2"
    exit 0
  fi
  shift
done
exit 1
''')

    def stub(self, name, body):
        path = self.bin / name
        path.write_text('#!/bin/bash\nset -eu\n' + body)
        path.chmod(0o755)

    def run_setup(self, *args):
        return subprocess.run(['bash', str(self.repo / 'setup.sh'), *args],
                              env=self.env, text=True, capture_output=True, timeout=20)

    def calls(self):
        return self.log.read_text() if self.log.exists() else ''

    def test_dry_run_is_read_only(self):
        result = self.run_setup('--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(list(self.home.iterdir()))
        self.assertEqual(self.calls(), '')

    def test_configure_only_and_repeat(self):
        for _ in range(2):
            result = self.run_setup('--configure-only')
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), '')
        self.assertFalse(list(self.home.rglob('*.backup-*')))
        model = yaml.safe_load((self.home / '.hermes/config.yaml').read_text())['model']
        self.assertEqual(model['provider'], 'custom')
        self.assertEqual(model['api_mode'], 'chat_completions')
        self.assertEqual(model['context_length'], 65536)
        subprocess.run(['bash', '-n', str(self.home / '.config/local-model-setup/path.sh')], check=True)
        if shutil.which('fish'):
            subprocess.run(['fish', '-n', str(self.home / '.config/fish/conf.d/local-model-setup.fish')], check=True)

    @unittest.skipUnless(Path('/etc/arch-release').exists() and os.geteuid() != 0,
                         'shell install simulation requires non-root Arch host')
    def test_install_flow(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertIn('pacman -Syu --needed', calls)
        self.assertIn('installer --channel stable --non-interactive', calls)
        self.assertEqual(calls.count('ollama pull '), 2)
        self.assertIn('hermes --version', calls)
        self.assertIn('verify', calls)
        self.assertNotIn('cline', calls)

    @unittest.skipUnless(Path('/etc/arch-release').exists() and os.geteuid() != 0,
                         'shell tuning simulation requires non-root Arch host')
    def test_tune_only_and_verification_failure(self):
        self.env['VERIFY_FAIL'] = '1'
        result = self.run_setup('--tune-only')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.home / '.hermes').exists())
        self.env['VERIFY_FAIL'] = '0'
        result = self.run_setup('--tune-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('pacman', self.calls())
        self.assertNotIn('ollama pull', self.calls())
        self.assertNotIn('installer', self.calls())

    @unittest.skipUnless(Path('/etc/arch-release').exists() and os.geteuid() != 0,
                         'shell install simulation requires non-root Arch host')
    def test_failed_install_verification_preserves_existing_agent_config(self):
        path = self.home / '.hermes/config.yaml'
        path.parent.mkdir()
        original = 'model: previous:model\n'
        path.write_text(original)
        self.env['VERIFY_FAIL'] = '1'
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('verify', self.calls())
        self.assertEqual(path.read_text(), original)
        self.assertFalse((self.home / '.ollama/config.json').exists())

    def test_custom_context_and_home_through_shell(self):
        custom = self.home / 'custom hermes profile'
        self.env['HERMES_HOME'] = str(custom)
        with (self.repo / 'config.sh').open('a') as f:
            f.write('\nCONTEXT_LENGTH=32768\n')
        result = self.run_setup('--configure-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        config = yaml.safe_load((custom / 'config.yaml').read_text())
        self.assertEqual(config['model']['context_length'], 32768)
        self.assertFalse((self.home / '.hermes').exists())
        self.assertEqual(self.calls(), '')

    @unittest.skipUnless(Path('/etc/arch-release').exists() and os.geteuid() != 0,
                         'shell install simulation requires non-root Arch host')
    def test_service_and_model_download_failures_stop_install(self):
        for command, body in (
            ('sudo', 'if [[ "$*" == "systemctl restart ollama.service" ]]; then exit 1; fi\n'
                     'if [[ "$1" == test ]]; then exit 1; fi\n'),
            ('ollama', 'exit 1\n'),
        ):
            with self.subTest(command=command):
                path = self.bin / command
                original = path.read_text()
                self.stub(command, body)
                self.log.unlink(missing_ok=True)
                try:
                    result = self.run_setup()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn('installer', self.calls())
                    self.assertNotIn('verify', self.calls())
                    self.assertFalse((self.home / '.hermes').exists())
                    self.assertNotIn('Setup complete.', result.stdout)
                finally:
                    path.write_text(original)

    def test_zero_keep_alive_is_rejected_before_changes(self):
        with (self.repo / 'config.sh').open('a') as f:
            f.write('\nKEEP_ALIVE="0s"\n')
        result = self.run_setup('--dry-run')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('KEEP_ALIVE', result.stderr)
        self.assertEqual(self.calls(), '')
        self.assertFalse(list(self.home.iterdir()))

    def test_invalid_arguments_and_duplicate_models(self):
        for args in [('--unknown',), ('--dry-run', 'extra')]:
            self.assertNotEqual(self.run_setup(*args).returncode, 0)
        with (self.repo / 'config.sh').open('a') as f:
            f.write('\nMODELS+=("${MODELS[0]}")\n')
        result = self.run_setup('--dry-run')
        self.assertIn('Duplicate model', result.stderr)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), '')

    @unittest.skipUnless(Path('/etc/arch-release').exists() and os.geteuid() != 0,
                         'shell install simulation requires non-root Arch host')
    def test_download_and_installer_failures_stop_setup(self):
        for flag in ('DOWNLOAD_FAIL', 'INSTALL_FAIL', 'PACMAN_FAIL'):
            with self.subTest(flag=flag):
                self.env[flag] = '1'
                self.log.unlink(missing_ok=True)
                result = self.run_setup()
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('hermes --version', self.calls())
                self.assertNotIn('verify', self.calls())
                self.assertFalse((self.home / '.hermes').exists())
                del self.env[flag]

    @unittest.skipUnless(Path('/etc/arch-release').exists() and os.geteuid() != 0,
                         'shell tuning simulation requires non-root Arch host')
    def test_invalid_config_fails_before_service_changes(self):
        (self.home / '.hermes').mkdir()
        (self.home / '.hermes/config.yaml').write_text('model: [bad')
        result = self.run_setup('--tune-only')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), '')


if __name__ == '__main__':
    unittest.main()
