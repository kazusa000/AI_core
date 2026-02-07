from __future__ import annotations

import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from src.asr.factory import ASR_REGISTRY
from src.llm.factory import LLM_REGISTRY
from src.tts.factory import TTS_REGISTRY

_START_LOCKS: dict[str, threading.Lock] = {}
_DEFAULT_STARTUP_TIMEOUT_S = 120.0


def is_endpoint_ready(endpoint: str, verify_ssl: bool, timeout_s: float = 3.0) -> bool:
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return False

    probe_url = f"{parsed.scheme}://{parsed.netloc}/"
    req = urllib.request.Request(probe_url, method="GET")

    ssl_ctx = None
    if parsed.scheme.lower() == "https":
        if verify_ssl:
            ssl_ctx = ssl.create_default_context()
        else:
            ssl_ctx = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ssl_ctx) as resp:
            return resp.status in (200, 204)
    except urllib.error.HTTPError as exc:
        return exc.code == 404
    except Exception:
        return False


def _registry_for_service_type(service_type: str):
    if service_type == "tts":
        return TTS_REGISTRY
    if service_type == "asr":
        return ASR_REGISTRY
    if service_type == "llm":
        return LLM_REGISTRY
    raise RuntimeError(f"Unsupported service_type: {service_type}")


def _resolve_start_script(service_type: str, model_name: str) -> Path:
    registry = _registry_for_service_type(service_type)
    entry = registry.get(model_name)
    if entry is None:
        raise RuntimeError(f"Unknown {service_type} model: {model_name}")

    ai_core_root = Path(__file__).resolve().parents[2]
    script_path = ai_core_root / entry.model_dir / "run_https.sh"
    if not script_path.is_file():
        raise RuntimeError(f"start script not found: {script_path}")
    return script_path


def ensure_remote_backend_ready(service_type: str, model_name: str, endpoint: str, verify_ssl: bool) -> None:
    if is_endpoint_ready(endpoint, verify_ssl=verify_ssl):
        return

    lock_key = f"{service_type}:{model_name}"
    lock = _START_LOCKS.setdefault(lock_key, threading.Lock())
    with lock:
        if is_endpoint_ready(endpoint, verify_ssl=verify_ssl):
            return

        ai_core_root = Path(__file__).resolve().parents[2]
        start_script = _resolve_start_script(service_type=service_type, model_name=model_name)
        log_path = Path(f"/tmp/{service_type}_{model_name}_autostart.log")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_fp:
            log_fp.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] auto-start {service_type}:{model_name} -> {start_script}\n".encode("utf-8")
            )
            subprocess.Popen(
                ["/bin/bash", str(start_script)],
                cwd=str(ai_core_root),
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        deadline = time.time() + _DEFAULT_STARTUP_TIMEOUT_S
        while time.time() < deadline:
            if is_endpoint_ready(endpoint, verify_ssl=verify_ssl):
                return
            time.sleep(1.0)

    raise RuntimeError(
        f"Remote backend '{service_type}:{model_name}' is not ready at {endpoint} after {_DEFAULT_STARTUP_TIMEOUT_S:.0f}s"
    )
