"""Verification fails on conflicting service settings before running inference."""
import contextlib
import io
import json
import unittest
from unittest.mock import MagicMock, patch

import verify


class VerificationTests(unittest.TestCase):
    def run_check(self, environment, context=65536, vram=1000, capabilities=None, generation=None, logs='library=ROCm ROCm0'):
        opener = MagicMock()
        responses = [
            {'capabilities': ['completion', 'tools'] if capabilities is None else capabilities},
            {'model': 'example:latest', 'done': True, 'response': 'Hello'} if generation is None else generation,
            {'models': [{'name': 'example:latest', 'size_vram': vram,
                         'context_length': context}]},
        ]
        opener.open.side_effect = [contextlib.closing(io.StringIO(json.dumps(r)))
                                   for r in responses]
        with patch('sys.argv', ['verify.py', '--model', 'example', '--context', '65536',
                                '--kv-cache', 'q8_0', '--flash-attention', '1']), \
             patch('verify.urllib.request.build_opener', return_value=opener), \
             patch('verify.subprocess.check_output', side_effect=[
                 environment, 'invocation-id', logs]), \
             contextlib.redirect_stdout(io.StringIO()):
            verify.main()
        return opener

    environment = ('OLLAMA_CONTEXT_LENGTH=65536 OLLAMA_KV_CACHE_TYPE=q8_0 '
                   'OLLAMA_FLASH_ATTENTION=1 OLLAMA_HOST=127.0.0.1:11434 '
                   'OLLAMA_VULKAN=0 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1')

    def test_matching_service_and_loaded_context_pass(self):
        self.assertEqual(self.run_check(self.environment).open.call_count, 3)

    def test_later_override_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, 'OLLAMA_CONTEXT_LENGTH'):
            self.run_check(self.environment.replace('65536', '262144'))

    def test_conflicting_service_runtime_settings_are_rejected(self):
        for key, old, new in (
            ('OLLAMA_HOST', '127.0.0.1:11434', '0.0.0.0:11434'),
            ('OLLAMA_VULKAN', '0', '1'),
            ('OLLAMA_NUM_PARALLEL', '1', '4'),
            ('OLLAMA_MAX_LOADED_MODELS', '1', '3'),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(SystemExit, key):
                self.run_check(self.environment.replace(f'{key}={old}', f'{key}={new}'))

    def test_model_context_override_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, 'loaded context=262144'):
            self.run_check(self.environment, context=262144)

    def test_cpu_only_model_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, 'GPU allocation was not detected'):
            self.run_check(self.environment, vram=0)

    def test_model_without_tools_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, 'tool calling'):
            self.run_check(self.environment, capabilities=['completion'])

    def test_empty_or_incomplete_generation_is_rejected(self):
        for generation in ({'done': True, 'response': ''}, {'response': 'partial'}):
            with self.assertRaisesRegex(SystemExit, 'nonempty generation'):
                self.run_check(self.environment, generation=generation)

    def test_api_error_stops_verification(self):
        with self.assertRaisesRegex(RuntimeError, 'model failed to load'):
            self.run_check(self.environment, generation={'error': 'model failed to load'})

    def test_other_gpu_backend_does_not_pass_rocm_check(self):
        with self.assertRaisesRegex(SystemExit, 'ROCm was not verified'):
            self.run_check(self.environment, logs='library=Vulkan Vulkan0')


if __name__ == '__main__':
    unittest.main()
