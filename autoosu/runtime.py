"""App-owned uv/Python runtime. The GUI can stay on its bundled CPU interpreter."""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from . import __version__
from .devices import CudaStatus, _nvidia_name, describe_cuda, detect_cuda
from .i18n import tr

_PROCESS_LOCK = threading.Lock()


class SetupCancelled(RuntimeError):
    pass


def runtime_root() -> Path:
    override = os.environ.get("AUTOOSU_RUNTIME_HOME")
    if override:
        return Path(override).resolve()
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()/".local"/"share")
    return base / "AUTO-OSU" / "runtime"


def child_environment() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith('_PYI_')}
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", AUTOOSU_MANAGED_WORKER="1",
               UV_PYTHON_INSTALL_DIR=str(runtime_root()/"python"), UV_NO_PROGRESS="1", UV_NO_CONFIG="1")
    return env


def popen(args, **kwargs):
    """Avoid a frozen GUI's DLL search path contaminating the external Python."""
    with _PROCESS_LOCK:
        frozen_windows = sys.platform == "win32" and getattr(sys, "frozen", False)
        if frozen_windows:
            import ctypes
            ctypes.windll.kernel32.SetDllDirectoryW(None)
        try:
            return subprocess.Popen([str(x) for x in args], env=child_environment(),
                                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                                    **kwargs)
        finally:
            if frozen_windows:
                ctypes.windll.kernel32.SetDllDirectoryW(str(sys._MEIPASS))


def capture(args, timeout=60):
    with popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace") as proc:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise RuntimeError("Runtime check timed out") from None
        if proc.returncode:
            raise RuntimeError((err or out)[-3500:] or f"Process exited with {proc.returncode}")
        return out


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise SetupCancelled("Setup cancelled; the previous runtime is unchanged")


def run_logged(args, log, cancel=None, *, timeout=3600, heartbeat_interval=15):
    """Keep quiet installers observable, cancellable and bounded, including after EOF."""
    _check_cancel(cancel)
    proc = popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                 start_new_session=sys.platform != "win32")
    messages = queue.Queue()
    def read():
        try:
            with proc.stdout:
                for line in proc.stdout:
                    messages.put(line.rstrip())
        finally:
            messages.put(None)
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    tail = []
    started = last_output = last_notice = time.monotonic()
    output_closed = False
    try:
        while not output_closed or proc.poll() is None:
            _check_cancel(cancel)
            now = time.monotonic()
            if now - started >= timeout:
                raise RuntimeError(tr('runtime.timeout', seconds=round(timeout)) + "\n" + "\n".join(tail))
            if now - max(last_output, last_notice) >= heartbeat_interval:
                log(tr('runtime.waiting', elapsed=int(now - started), quiet=int(now - last_output)))
                last_notice = now
            try:
                line = messages.get(timeout=.15)
            except queue.Empty:
                continue
            if line is None:
                output_closed = True
            elif line:
                last_output = time.monotonic()
                tail.append(line)
                tail = tail[-20:]
                log(line)
        code = proc.wait()
        _check_cancel(cancel)
        if code:
            raise RuntimeError("\n".join(tail) or f"Installer exited with {code}")
    finally:
        if proc.poll() is None or not output_closed:
            if sys.platform == "win32" and proc.poll() is None:
                try:
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW, timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            elif sys.platform != "win32":
                # The build backend may hold the output pipe open after uv exits.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=15)
        reader.join(timeout=2)


def _download(url, cancel=None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": f"AUTO-OSU/{__version__}"})
    chunks = []
    with urllib.request.urlopen(req, timeout=45) as response:
        while True:
            _check_cancel(cancel)
            chunk = response.read(1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def _uv_usable(path) -> bool:
    try:
        help_text = capture([path, "pip", "install", "--help"], timeout=15)
        # Older uv versions can miss modern CUDA backends even when they support auto.
        return "--torch-backend" in help_text and "cu130" in help_text
    except (OSError, RuntimeError):
        return False


def ensure_uv(log=print, cancel=None) -> Path:
    exe = "uv.exe" if sys.platform == "win32" else "uv"
    owned = runtime_root()/"tools"/exe
    candidates = [owned]
    system = shutil.which("uv")
    if system:
        candidates.append(Path(system))
    for path in candidates:
        if path.is_file() and _uv_usable(path):
            return path
    log("Downloading uv from its official PyPI distribution…")
    metadata = json.loads(_download("https://pypi.org/pypi/uv/json", cancel))
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    def supported(name):
        if sys.platform == "win32":
            return name.endswith("win_arm64.whl" if arm else "win_amd64.whl")
        if sys.platform == "darwin":
            return "macosx" in name and name.endswith("arm64.whl" if arm else "x86_64.whl")
        return "manylinux" in name and name.endswith("aarch64.whl" if arm else "x86_64.whl")
    asset = next((f for f in metadata['urls'] if supported(f['filename'])), None)
    if asset is None:
        raise RuntimeError("No uv binary is available for this platform; install uv and retry")
    data = _download(asset['url'], cancel)
    if hashlib.sha256(data).hexdigest() != asset['digests']['sha256']:
        raise RuntimeError("uv download checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        member = next(n for n in archive.namelist() if Path(n).name == exe)
        binary = archive.read(member)
    owned.parent.mkdir(parents=True, exist_ok=True)
    temporary = owned.with_suffix('.download')
    temporary.write_bytes(binary)
    temporary.chmod(0o755)
    temporary.replace(owned)
    if not _uv_usable(owned):
        raise RuntimeError("The downloaded uv does not support automatic CUDA setup")
    log(capture([owned, "--version"]).strip())
    return owned


def python_in(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _owned_python() -> Path | None:
    """Use the real patch-version directory, not uv's Windows minor-version junction."""
    root = runtime_root()/"python"
    for folder in sorted(root.glob('cpython-3.12.*'), reverse=True):
        executable = folder/("python.exe" if sys.platform == "win32" else "bin/python3")
        if not executable.is_file():
            continue
        try:
            result = capture([executable, '-I', '-c', 'import sys; print(sys.version_info[:2])'], timeout=15)
            if result.strip() == '(3, 12)':
                return executable
        except (OSError, RuntimeError):
            pass
    return None


def create_environment(uv: Path, environment: Path, log, cancel=None) -> None:
    interpreter = _owned_python()
    if interpreter:
        run_logged([uv, 'venv', '--python', interpreter, '--no-python-downloads', environment], log, cancel)
        return
    try:
        run_logged([uv, 'venv', '--python', '3.12', '--managed-python', environment], log, cancel)
    except RuntimeError as exc:
        # uv #19622: Windows may fail after successfully downloading Python because
        # the minor-version junction does not resolve. Do not touch system links.
        if 'Python minor version link' not in str(exc):
            raise
        interpreter = _owned_python()
        if interpreter is None:
            raise
        log("Using the downloaded Python directly; its Windows version shortcut was unavailable.")
        run_logged([uv, 'venv', '--python', interpreter, '--no-python-downloads', environment], log, cancel)


def active_python() -> Path | None:
    try:
        data = json.loads((runtime_root()/"active.json").read_text(encoding='utf-8'))
        if data.get('app_version') != __version__:
            return None
        path = Path(data['python']).resolve()
        if path.is_file() and path.is_relative_to((runtime_root()/"envs").resolve()):
            return path
    except (OSError, ValueError, KeyError):
        pass
    return None


def probe_python(python: Path) -> CudaStatus:
    script = ("import json; from dataclasses import asdict; from autoosu.devices import detect_cuda; "
              "from autoosu.worker import main; print(json.dumps(asdict(detect_cuda())))")
    return CudaStatus(**json.loads(capture([python, '-I', '-c', script], timeout=90).strip()))


@dataclass
class RuntimeStatus:
    cuda: CudaStatus
    python: Path | None = None
    managed_error: str = ""


def detect_runtime() -> RuntimeStatus:
    python = active_python()
    error = ""
    if python:
        try:
            status = probe_python(python)
            if status.available:
                return RuntimeStatus(status, python)
            error = describe_cuda(status)
        except Exception as exc:
            error = str(exc)
    return RuntimeStatus(detect_cuda(), managed_error=error)


@contextmanager
def installation_lock():
    root = runtime_root()
    root.mkdir(parents=True, exist_ok=True)
    # OS-backed lock is released after a crash; no stale lock file can block retry.
    with open(root/'install.lock', 'a+b') as handle:
        handle.seek(0)
        handle.write(b'0')
        handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Another AUTO-OSU window is already setting up the runtime") from None
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _source_for_install() -> tuple[Path, bool]:
    if not getattr(sys, "frozen", False):
        source = Path(__file__).resolve().parents[1]
        if (source/'pyproject.toml').is_file():
            return source, True
    bundle = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))/'runtime'/'runtime-source.zip'
    if not bundle.is_file():
        raise RuntimeError("This build does not include the runtime source bundle; use an updated AUTO-OSU build")
    return bundle, False


def install_runtime(*, progress=None, log=print, cancel=None) -> RuntimeStatus:
    progress = progress or (lambda fraction, message: None)
    with installation_lock():
        _check_cancel(cancel)
        if not _nvidia_name() and not detect_cuda().available:
            raise RuntimeError("NVIDIA GPU/driver was not detected. Install the NVIDIA driver, then retry GPU setup.")
        progress(.04, 'runtime.uv')
        uv = ensure_uv(log, cancel)
        environment = runtime_root()/'envs'/f"cuda-{uuid.uuid4().hex[:12]}"
        progress(.12, 'runtime.python')
        create_environment(uv, environment, log, cancel)
        python = python_in(environment)
        source, editable = _source_for_install()
        progress(.28, 'runtime.dependencies')
        log(tr('runtime.dependencies_hint'))
        args = [uv, 'pip', 'install', '--python', python, '--torch-backend', 'auto', '--index-url', 'https://pypi.org/simple']
        if editable:
            args.append('--editable')
        args += [source, 'imageio-ffmpeg']
        run_logged(args, log, cancel)
        progress(.92, 'runtime.verify')
        status = probe_python(python)
        _check_cancel(cancel)
        if not status.available:
            raise RuntimeError(describe_cuda(status))
        manifest = dict(python=str(python), app_version=__version__, source=str(source),
                        installed_at=time.time(), cuda=asdict(status))
        path = runtime_root()/'active.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        temporary.replace(path)
        progress(1.0, 'runtime.ready')
        return RuntimeStatus(status, python)


def run_in_runtime(python: Path, request: dict, on_event, cancel=None) -> None:
    jobs = runtime_root()/'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    spec = jobs/f'{uuid.uuid4().hex}.json'
    stop = spec.with_suffix('.cancel')
    request['cancel_file'] = str(stop)
    spec.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
    final_event = None
    error = ''
    proc = popen([python, '-I', '-m', 'autoosu.worker', spec], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                 text=True, encoding='utf-8', errors='replace')
    def signal_stop():
        while proc.poll() is None:
            if cancel is not None and cancel.wait(.15):
                stop.touch()
                return
            if cancel is None:
                time.sleep(.15)
    watcher = threading.Thread(target=signal_stop, daemon=True)
    watcher.start()
    try:
        for line in proc.stdout:
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or 'event' not in event:
                    raise ValueError()
            except ValueError:
                on_event({'event': 'log', 'text': line.rstrip()})
                continue
            if event['event'] == 'error':
                error = event['text']
            else:
                if event['event'] in ('single_done', 'batch_done'):
                    final_event = event
                else:
                    on_event(event)
        code = proc.wait()
        if code or error or final_event is None:
            raise RuntimeError(error or f"Inference worker exited without a result (exit {code})")
        on_event(final_event)
    finally:
        if proc.poll() is None:
            stop.touch()
            proc.wait()
        watcher.join(timeout=2)
        proc.stdout.close()
        spec.unlink(missing_ok=True)
        stop.unlink(missing_ok=True)
