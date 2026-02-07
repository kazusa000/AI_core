from __future__ import annotations

import argparse
import os

import uvicorn

SERVICE_IMPORTS = {
    "asr": "services.asr_service:app",
    "tts": "services.tts_service:app",
    "llm": "services.llm_service:app",
    "recorder": "services.recorder_service:app",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ai_core FastAPI service with optional HTTPS")
    parser.add_argument("service", choices=sorted(SERVICE_IMPORTS.keys()))
    parser.add_argument("--host", default=os.environ.get("AI_CORE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AI_CORE_PORT", "8443")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--ssl-certfile", default=os.environ.get("AI_CORE_SSL_CERTFILE"))
    parser.add_argument("--ssl-keyfile", default=os.environ.get("AI_CORE_SSL_KEYFILE"))

    args = parser.parse_args()

    app_ref = SERVICE_IMPORTS[args.service]
    uvicorn.run(
        app_ref,
        host=args.host,
        port=args.port,
        reload=args.reload,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )


if __name__ == "__main__":
    main()
