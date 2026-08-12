"""Hash the exported GGUF where it actually lives, to settle provenance disputes.

`modal volume get` pulls 5 GB over a home connection; the Hub copy is uploaded
from inside the container. When those two disagree, the Volume is the referee —
without it there is no way to tell which copy drifted, and a benchmark run
against a corrupted local file would silently report a wrong score.

    uv run --with modal modal run finetune/modal_hash_gguf.py
"""
from pathlib import Path

import modal

OUTPUTS_DIR = "/outputs"
GGUF_NAME = "qwen3-8b-wallet-ft.Q4_K_M.gguf"

image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("qwen-hash")
outputs = modal.Volume.from_name("qwen-ft-outputs", create_if_missing=True)


@app.function(image=image, timeout=1800, volumes={OUTPUTS_DIR: outputs})
def sha256() -> str:
    import hashlib

    path = Path(OUTPUTS_DIR) / GGUF_NAME
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return f"{h.hexdigest()}  {path.stat().st_size} bytes"


@app.local_entrypoint()
def main() -> None:
    print("volume:", sha256.remote())
