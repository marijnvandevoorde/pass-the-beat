"""Put the beat-this `final0` checkpoint in the torch hub cache, so
`Audio2Beats(checkpoint_path="final0")` loads it without a download.

Tries the mirror (a release asset on this repo) first, then the original
host (cloud.cp.jku.at, down since 2026-10-21), and accepts a file only if its
sha256 matches. Used by the Dockerfile and CI: `python scripts/fetch-checkpoint.py`.
"""

import hashlib
import os
import sys
import urllib.request

import torch

SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
URLS = [
    "https://github.com/marijnvandevoorde/pass-the-beat/releases/download/checkpoint-final0/final0.ckpt",
    "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt",
]
# The file name beat_this asks torch hub for when given the name "final0".
TARGET = os.path.join(torch.hub.get_dir(), "checkpoints", "beat_this-final0.ckpt")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if os.path.exists(TARGET) and sha256(TARGET) == SHA256:
        print("checkpoint already cached:", TARGET)
        return
    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    tmp = TARGET + ".part"
    for url in URLS:
        try:
            print("downloading", url)
            urllib.request.urlretrieve(url, tmp)
        except OSError as e:
            print("  failed:", e)
            continue
        got = sha256(tmp)
        if got == SHA256:
            os.replace(tmp, TARGET)
            print("checkpoint verified:", TARGET)
            return
        print("  sha256 mismatch:", got)
    if os.path.exists(tmp):
        os.remove(tmp)
    sys.exit("no source gave a checkpoint with sha256 " + SHA256)


if __name__ == "__main__":
    main()
