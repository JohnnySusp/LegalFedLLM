import ast
import time
from types import SimpleNamespace
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from desktop.app import _host_preview_terminal_failure, _host_preview_finished
from desktop.profiles import PortableProfileManager
from client.peft_backend import TransformersPeftTrainingBackend
from client.runtime import ClientRuntime
from client.main import CoordinatorGateway, create_app
from client.tunnel import SshTunnelConfig, SshTunnelManager
from tests.test_client_desktop_api import mock_profile
import httpx


class PersistenceTests(unittest.TestCase):
    def test_restart_and_reset_preserve_explicit_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"APPIMAGE": str(Path(directory)/"Client.AppImage"), "LEGALFEDLLM_DATA_ROOT": ""}):
                manager = PortableProfileManager()
                self.assertTrue(manager.desktop_settings()["low_vram_mode"])
                manager.set_desktop_setting("low_vram_mode", False)
                restarted = PortableProfileManager()
                self.assertFalse(restarted.desktop_settings()["low_vram_mode"])
                restarted.reset_desktop_settings()
                self.assertTrue(PortableProfileManager().desktop_settings()["low_vram_mode"])

    def test_rejection_survives_restart_and_is_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            manager.reject_reverse_preview("client-a", "round-5")
            manager.set_desktop_setting("low_vram_mode", False)
            manager = PortableProfileManager(directory)
            self.assertTrue(manager.reverse_preview_rejected("client-a", "round-5"))
            self.assertFalse(manager.reverse_preview_rejected("client-b", "round-5"))
            self.assertFalse(manager.reverse_preview_rejected("client-a", "round-6"))
            self.assertTrue(_host_preview_terminal_failure("HTTP 409: Host package timestamp is outside the allowed skew"))
            self.assertFalse(_host_preview_terminal_failure("HTTP 409: another local ML job is already running"))

    def test_preview_failure_shows_terminal_error_once_and_backs_off_transient(self):
        source = (Path(__file__).parents[1] / 'desktop/app.py').read_text()
        tree = ast.parse(source)
        node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_host_preview_failed')
        namespace = dict(_host_preview_finished=_host_preview_finished, _host_preview_terminal_failure=_host_preview_terminal_failure, time=time)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<preview>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            window = SimpleNamespace(manager=PortableProfileManager(directory), profile=SimpleNamespace(profile_id='a'), previewed_rounds=set(), host_preview_inflight={'r'}, host_preview_retry_at={}, message=mock.Mock(), _show_error=mock.Mock())
            namespace['_host_preview_failed'](window, 'r', 'HTTP 409: another local ML job is already running')
            window._show_error.assert_not_called()
            self.assertGreater(window.host_preview_retry_at['r'], time.monotonic())
            self.assertNotIn('r', window.previewed_rounds)
            namespace['_host_preview_failed'](window, 'r', 'HTTP 409: Host package timestamp is outside the allowed skew')
            window._show_error.assert_called_once()
            self.assertIn('r', window.previewed_rounds)
            self.assertTrue(PortableProfileManager(directory).reverse_preview_rejected('a', 'r'))

    def test_corrupt_state_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            manager.state_path.write_text('{broken')
            with self.assertRaises(ValueError):
                manager.set_desktop_setting("low_vram_mode", True)
            self.assertEqual(manager.state_path.read_text(), '{broken')


class ServingTests(unittest.TestCase):
    def test_session_reused_evicted_and_reloaded_for_checkpoint(self):
        backend = object.__new__(TransformersPeftTrainingBackend)
        backend.execution_profile = mock.Mock(device="cpu")
        backend.model_profile = mock.Mock()
        backend.checkpoints = mock.Mock()
        backend.checkpoints.current.return_value = None
        torch = mock.MagicMock()
        torch.cuda.is_available.return_value = False
        tokenizer = mock.Mock()
        tokenizer.decode.return_value = 'answer'
        backend._dependencies = mock.Mock(return_value=(torch, mock.Mock(), mock.MagicMock(), None))
        backend._validate_device = mock.Mock()
        backend._load_tokenizer = mock.Mock(return_value=tokenizer)
        backend._load_base_model = mock.Mock(return_value=mock.MagicMock())
        backend._verify_loaded_adapter = mock.Mock()
        with mock.patch('client.peft_backend.encode_chat_prompt', return_value=[1, 2]):
            for _ in range(2):
                self.assertEqual(backend.generate_text([], 5), 'answer')
            self.assertEqual(backend._load_base_model.call_count, 1)
            backend.checkpoints.current.return_value = ('v2', '/adapter')
            backend.generate_text([], 5)
            self.assertEqual(backend._load_base_model.call_count, 2)
            backend.release_serving_session()
            self.assertIsNone(backend._serving_session)
            backend.generate_text([], 5)
            self.assertEqual(backend._load_base_model.call_count, 3)


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_exclusive_job_evicts_serving_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = ClientRuntime(data_dir=directory, client_id='client-test', model_profile=mock_profile())
            events = []
            runtime.release_serving_session = mock.Mock(side_effect=lambda: events.append('evicted'))
            runtime.local_train = mock.Mock(side_effect=lambda examples: events.append('trained') or {})
            app = create_app(runtime, CoordinatorGateway('http://unused', None), admin_token_override='secret', tunnel_manager=SshTunnelManager(SshTunnelConfig(enabled=False, target='')))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://client', headers={'X-Client-Admin-Token':'secret'}) as client:
                response = await client.post('/v1/local-train', json={'examples':['sample']})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(events, ['evicted', 'trained'])
                async with app.state.ml_lock:
                    response = await client.post('/v1/local-train', json={'examples':['sample']})
                self.assertEqual(response.status_code, 409)
                self.assertIn('already running', response.text)
                self.assertEqual(events, ['evicted', 'trained'])

    async def test_local_inference_busy_message_is_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = mock_profile().model_copy(
                update={"serving_backend": "transformers"}
            )
            runtime = ClientRuntime(
                data_dir=directory,
                client_id="client-test",
                model_profile=profile,
            )
            runtime.generate_transformers = mock.Mock(return_value="LOCAL")
            app = create_app(
                runtime,
                CoordinatorGateway("http://unused", None),
                admin_token_override="secret",
                tunnel_manager=SshTunnelManager(
                    SshTunnelConfig(enabled=False, target="")
                ),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client",
                headers={"Authorization": "Bearer secret"},
            ) as client:
                async with app.state.ml_lock:
                    response = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "legalfedllm-local",
                            "messages": [
                                {"role": "user", "content": "test"}
                            ],
                        },
                    )

            self.assertEqual(response.status_code, 409, response.text)
            self.assertIn(
                "LOCAL inference is temporarily unavailable",
                response.text,
            )

    async def test_same_history_switches_both_directions_and_logs_only_model(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = ClientRuntime(data_dir=directory, client_id='client-test', model_profile=mock_profile())
            gateway = CoordinatorGateway('http://unused', None)
            gateway.generate_host = mock.AsyncMock(return_value={'text': 'HOST'})
            runtime.generate = mock.AsyncMock(return_value='LOCAL')
            app = create_app(runtime, gateway, admin_token_override='secret', tunnel_manager=SshTunnelManager(SshTunnelConfig(enabled=False, target='')))
            history = [{'role': 'user', 'content': 'private prompt'}]
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://client', headers={'Authorization':'Bearer secret'}) as client:
                with self.assertLogs('uvicorn.error', level='INFO') as logs:
                    for model in ['local', 'host', 'local', 'host']:
                        response = await client.post('/v1/chat/completions', json={'model':f'legalfedllm-{model}', 'messages':history})
                        self.assertEqual(response.status_code, 200, response.text)
                        answer = response.json()['choices'][0]['message']
                        self.assertEqual(answer['content'], model.upper())
                        history.extend([answer, {'role':'user', 'content':'private next prompt'}])
                self.assertNotIn('private', ' '.join(logs.output))
                self.assertEqual(gateway.generate_host.await_count, 2)
