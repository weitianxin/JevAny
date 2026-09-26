# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""FastAPI server for prefill-only decisions.

Run: uv run --extra serve python -m jevany.serve --run runs/rlcr --port 8008

TypeSafe-compatible: POST /v1/systemone and GET /v1/models (no auth). JEVANY_PREFIX_CACHE /
JEVANY_PREFIX_MIN_TOKENS size the state-prefix cache; JEVANY_DATE_FACTS=1 enables deterministic date preprocessing.
"""
import argparse, os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .api import SystemOneRequest, with_date_facts
from .runtime import (DEFAULT_CHECKPOINT, DecisionRuntime, JevModel, INFER_MAX_STATE,
                      INFER_MAX_BRANCH, INFER_MAX_PACKED, PREFIX_CACHE_SIZE, PREFIX_MIN_TOKENS)
DATE_FACTS = os.environ.get("JEVANY_DATE_FACTS", "0") == "1"
MEDIA_ROOT = os.environ.get("JEVANY_MEDIA_ROOT")
MEDIA_MAX_BYTES = int(os.environ.get("JEVANY_MEDIA_MAX_BYTES", str(50 * 1024 * 1024)))
MEDIA_TOTAL_BYTES = int(os.environ.get("JEVANY_MEDIA_TOTAL_BYTES", str(100 * 1024 * 1024)))
MEDIA_MAX_PIXELS = int(os.environ.get("JEVANY_MEDIA_MAX_PIXELS", str(4096 * 4096)))
MEDIA_MAX_VIDEO_FRAMES = int(os.environ.get("JEVANY_MEDIA_MAX_VIDEO_FRAMES", "3600"))
if min(MEDIA_MAX_BYTES, MEDIA_TOTAL_BYTES, MEDIA_MAX_PIXELS, MEDIA_MAX_VIDEO_FRAMES) <= 0:
    raise ValueError("JEVANY media limits must be positive")
if MEDIA_TOTAL_BYTES < MEDIA_MAX_BYTES:
    raise ValueError("JEVANY_MEDIA_TOTAL_BYTES must be at least JEVANY_MEDIA_MAX_BYTES")


class Server(DecisionRuntime):
    """The loaded checkpoint and the state-prefix cache shared by every request (one model, one lock)."""
    def answer(self, req):
        """The /v1/systemone response body for one request."""
        try:
            return super().answer(prepare(req))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error


def _validate_media_file(item, path):
    try:
        if item.type == "image":
            from PIL import Image
            with Image.open(path) as image:
                if image.width * image.height > MEDIA_MAX_PIXELS:
                    raise HTTPException(422, f"image exceeds {MEDIA_MAX_PIXELS} pixels")
                image.verify()
        else:
            import av
            with av.open(str(path)) as container:
                streams = container.streams.video
                if not streams:
                    raise HTTPException(422, "video file has no video stream")
                stream = streams[0]
                if stream.width * stream.height > MEDIA_MAX_PIXELS:
                    raise HTTPException(422, f"video frame exceeds {MEDIA_MAX_PIXELS} pixels")
                if not stream.frames:
                    raise HTTPException(422, "video frame count is unavailable")
                if stream.frames > MEDIA_MAX_VIDEO_FRAMES:
                    raise HTTPException(422, f"video exceeds {MEDIA_MAX_VIDEO_FRAMES} frames")
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(422, "install the multimodal extra to serve media")
    except Exception:
        raise HTTPException(422, "media file could not be validated")


def prepare(req):
    """Opt-in preprocessing applied to every request before the model sees it."""
    updates = {}
    if DATE_FACTS:
        updates["state"] = with_date_facts(req.state)
    if req.media:
        if not MEDIA_ROOT:
            raise HTTPException(422, "media is disabled; the server operator must set JEVANY_MEDIA_ROOT")
        try:
            root = Path(MEDIA_ROOT).resolve(strict=True)
        except (OSError, RuntimeError):
            raise HTTPException(422, "configured media root is unavailable")
        if not root.is_dir():
            raise HTTPException(422, "configured media root is not a directory")
        resolved, total = [], 0
        for item in req.media:
            if "://" in item.uri:
                raise HTTPException(422, "network media URIs are not allowed")
            candidate = Path(item.uri)
            try:
                path = candidate.resolve(strict=True) if candidate.is_absolute() else (root / candidate).resolve(strict=True)
            except (OSError, RuntimeError):
                raise HTTPException(422, "media file does not exist")
            try:
                path.relative_to(root)
            except ValueError:
                raise HTTPException(422, "media path is outside JEVANY_MEDIA_ROOT")
            if not path.is_file():
                raise HTTPException(422, "media path is not a regular file")
            size = path.stat().st_size
            if size > MEDIA_MAX_BYTES:
                raise HTTPException(422, f"media file exceeds {MEDIA_MAX_BYTES} bytes")
            total += size
            if total > MEDIA_TOTAL_BYTES:
                raise HTTPException(422, f"media files exceed {MEDIA_TOTAL_BYTES} bytes in total")
            _validate_media_file(item, path)
            resolved.append(item.model_copy(update={"uri": str(path)}))
        updates["media"] = resolved
    return req.model_copy(update=updates) if updates else req


app = FastAPI(title="JevAny")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def server() -> Server:
    return app.state.server


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    """TypeSafe-compatible endpoint: typed questions in, typed answers out, one prefill pass."""
    return server().answer(req)


@app.get("/v1/models")
def models():
    s = server()
    return {"models": [{"id": getattr(s, "model_id", "jevany-27b"), "aliases": ["jevany-latest"], "run": s.checkpoint.requested, "base": s.checkpoint.meta.base,
                        "lora": s.checkpoint.meta.lora, "device": s.device, "temperature": s.model.head.temperature,
                        "limits": {"state_tokens": INFER_MAX_STATE, "branch_tokens": INFER_MAX_BRANCH,
                                   "packed_tokens": INFER_MAX_PACKED,
                                   "media_enabled": bool(MEDIA_ROOT),
                                   "media_max_file_bytes": MEDIA_MAX_BYTES,
                                   "media_max_total_bytes": MEDIA_TOTAL_BYTES,
                                   "media_max_pixels": MEDIA_MAX_PIXELS,
                                   "media_max_video_frames": MEDIA_MAX_VIDEO_FRAMES},
                        "prefix_cache": {"size": PREFIX_CACHE_SIZE, "min_state_tokens": PREFIX_MIN_TOKENS, "hits": s.prefix_hits,
                                         "misses": s.prefix_misses, "cached_states": len(s.prefix_cache)}}]}


@app.get("/health")
def health():
    return {"status": "ready" if getattr(app.state, "server", None) is not None else "loading"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", "--run", dest="run", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--model-name", help="identity reported in every response; defaults to the checkpoint name")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    a = ap.parse_args(argv)
    local = JevModel.from_pretrained(a.run, device=a.device, dtype=a.dtype, model_name=a.model_name)
    runtime = local.runtime
    app.state.server = Server(runtime.checkpoint, runtime.tok, runtime.model, runtime.device, runtime.model_id)
    print(f"serving {runtime.checkpoint.requested} on {runtime.device} {a.host}:{a.port}")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
