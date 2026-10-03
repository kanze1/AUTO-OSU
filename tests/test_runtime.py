import json
from pathlib import Path
import sys
from threading import Event

import pytest

from autoosu import __version__, runtime
from autoosu.devices import CudaStatus
from autoosu.i18n import tr


@pytest.fixture
def owned(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOOSU_RUNTIME_HOME', str(tmp_path/'runtime'))
    return runtime.runtime_root()


def test_active_runtime_is_app_owned_and_versioned(owned):
    python = owned/'envs/gpu/Scripts/python.exe'
    python.parent.mkdir(parents=True)
    python.touch()
    manifest = owned/'active.json'
    manifest.write_text(json.dumps(dict(python=str(python), app_version=__version__)))
    assert runtime.active_python() == python
    manifest.write_text(json.dumps(dict(python=str(python), app_version='old')))
    assert runtime.active_python() is None
    other = owned/'system-python.exe'
    other.touch()
    manifest.write_text(json.dumps(dict(python=str(other), app_version=__version__)))
    assert runtime.active_python() is None


def fake_installer(monkeypatch):
    monkeypatch.setattr(runtime, '_nvidia_name', lambda: 'GPU')
    monkeypatch.setattr(runtime, 'ensure_uv', lambda *a: Path('uv'))
    monkeypatch.setattr(runtime, 'create_environment', lambda *a: None)
    monkeypatch.setattr(runtime, 'run_logged', lambda *a: None)
    monkeypatch.setattr(runtime, '_source_for_install', lambda: (Path('source'), True))


def test_failed_verification_preserves_active_runtime(owned, monkeypatch):
    owned.mkdir(parents=True)
    manifest = owned/'active.json'
    original = '{"previous":"working runtime"}'
    manifest.write_text(original)
    fake_installer(monkeypatch)
    monkeypatch.setattr(runtime, 'probe_python', lambda _: CudaStatus(False, 'runtime_error', detail='kernel failed'))
    with pytest.raises(RuntimeError, match='kernel failed'):
        runtime.install_runtime(log=lambda _: None)
    assert manifest.read_text() == original


def test_activate_only_after_cuda_verified(owned, monkeypatch):
    fake_installer(monkeypatch)
    def probe(python):
        assert not (owned/'active.json').exists()
        return CudaStatus(True, 'ready', 'GPU', 8, '13.0', 'test')
    monkeypatch.setattr(runtime, 'probe_python', probe)
    result = runtime.install_runtime(log=lambda _: None)
    manifest = json.loads((owned/'active.json').read_text())
    assert manifest['python'] == str(result.python)
    assert manifest['cuda']['available'] is True


def test_cancel_does_not_activate(owned):
    stop = Event()
    stop.set()
    with pytest.raises(runtime.SetupCancelled):
        runtime.install_runtime(cancel=stop)
    assert not (owned/'active.json').exists()


def test_minor_link_failure_uses_owned_interpreter(owned, monkeypatch):
    checks = iter([None, owned/'python/cpython-3.12.14/python.exe'])
    monkeypatch.setattr(runtime, '_owned_python', lambda: next(checks))
    calls = []
    def run(args, log, cancel):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError('Missing expected target directory for Python minor version link')
    monkeypatch.setattr(runtime, 'run_logged', run)
    runtime.create_environment(Path('uv'), owned/'envs/new', lambda _: None)
    assert '--managed-python' in calls[0]
    assert '--no-python-downloads' in calls[1]
    assert str(owned/'python/cpython-3.12.14/python.exe') == str(calls[1][3])


def test_child_environment_does_not_leak_python_paths(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', 'unexpected')
    monkeypatch.setenv('PYTHONHOME', 'unexpected')
    monkeypatch.setenv('_PYI_ARCHIVE_FILE', 'parent.exe')
    env = runtime.child_environment()
    assert 'PYTHONPATH' not in env and 'PYTHONHOME' not in env and '_PYI_ARCHIVE_FILE' not in env
    assert env['AUTOOSU_MANAGED_WORKER'] == '1'


def test_quiet_installer_reports_waiting_and_drains_output():
    logs = []
    runtime.run_logged([sys.executable, '-I', '-u', '-c',
                        "import time; print('Built runtime-source.zip'); time.sleep(.7); print('Installed')"],
                       logs.append, timeout=10, heartbeat_interval=.1)
    assert 'Built runtime-source.zip' in logs
    assert logs[-1] == 'Installed'
    assert logs[logs.index('Built runtime-source.zip') + 1:-1]


def test_precancelled_installer_does_not_start(monkeypatch):
    def unexpected_start(*args, **kwargs):
        pytest.fail('A cancelled installation must not start a subprocess')
    monkeypatch.setattr(runtime, 'popen', unexpected_start)
    cancel = Event()
    cancel.set()
    with pytest.raises(runtime.SetupCancelled):
        runtime.run_logged(['unused'], lambda _: None, cancel)


@pytest.mark.parametrize('close_output', [False, True])
@pytest.mark.parametrize('stop', ['cancel', 'timeout'])
def test_quiet_installer_can_stop_even_after_output_closes(tmp_path, monkeypatch, close_output, stop):
    processes = []
    real_popen = runtime.popen
    def tracked_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        processes.append(proc)
        real_wait = proc.wait
        def bounded_wait(timeout=None):
            assert timeout is not None or proc.poll() is not None, 'Uncancellable wait after output EOF'
            return real_wait(timeout=timeout)
        monkeypatch.setattr(proc, 'wait', bounded_wait)
        return proc
    monkeypatch.setattr(runtime, 'popen', tracked_popen)
    ready = tmp_path/'ready'
    script = "import os, pathlib, sys, time; print('Built runtime-source.zip', flush=True); "
    if close_output:
        script += "os.close(1); os.close(2); "
    script += "pathlib.Path(sys.argv[1]).touch(); time.sleep(20)"
    cancel = Event()
    logs = []
    def log(message):
        logs.append(message)
        if stop == 'cancel' and ready.exists() and message != 'Built runtime-source.zip':
            cancel.set()
    error = runtime.SetupCancelled if stop == 'cancel' else RuntimeError
    timeout = 10 if stop == 'cancel' else 3
    with pytest.raises(error) as exc:
        runtime.run_logged([sys.executable, '-I', '-u', '-c', script, ready], log, cancel,
                           timeout=timeout, heartbeat_interval=.1)
    assert ready.exists()
    assert 'Built runtime-source.zip' in logs
    assert all(proc.poll() is not None for proc in processes)
    if stop == 'timeout':
        assert tr('runtime.timeout', seconds=timeout) in str(exc.value)
        assert 'Built runtime-source.zip' in str(exc.value)


def test_installer_failure_keeps_original_diagnostic():
    with pytest.raises(RuntimeError, match='download connection failed'):
        runtime.run_logged([sys.executable, '-I', '-c',
                            "import sys; print('download connection failed'); sys.exit(2)"],
                           lambda _: None, timeout=10)


def test_install_explains_source_build_message(owned, monkeypatch):
    fake_installer(monkeypatch)
    monkeypatch.setattr(runtime, 'probe_python', lambda _: CudaStatus(True, 'ready'))
    logs = []
    runtime.install_runtime(log=logs.append)
    assert tr('runtime.dependencies_hint') in logs


@pytest.mark.parametrize('error', [runtime.SetupCancelled('cancelled'), RuntimeError('timeout')])
def test_interrupted_install_preserves_active_runtime(owned, monkeypatch, error):
    owned.mkdir(parents=True)
    manifest = owned/'active.json'
    original = '{"previous":"working runtime"}'
    manifest.write_text(original)
    fake_installer(monkeypatch)
    def interrupted(*args):
        raise error
    monkeypatch.setattr(runtime, 'run_logged', interrupted)
    with pytest.raises(type(error)):
        runtime.install_runtime(log=lambda _: None)
    assert manifest.read_text() == original
