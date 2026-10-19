"""End-to-end smoke test: boot the app (loads the beat-this model), then
analyse a 20 s, 120 BPM click track. Needs ffmpeg and network access for the
checkpoint download. Run from the repo root: `python tests/smoke.py`."""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    wav = os.path.join(d, "click.wav")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         "aevalsrc='if(lt(mod(t,0.5),0.02),sin(2*PI*1000*t),0)':s=22050:d=20",
         wav],
        check=True,
    )
    with TestClient(server.app) as client:
        health = client.get("/healthz").json()
        print("healthz:", health)
        assert health["model_loaded"], health

        with open(wav, "rb") as f:
            r = client.post("/analyze-audio", content=f.read())
        assert r.status_code == 200, r.text
        body = r.json()
        print("bpm:", body["bpm"], "beats:", len(body["beats"]))
        assert len(body["beats"]) > 10, body
        assert body["bpm"] is not None and abs(body["bpm"] - 120) < 5, body
