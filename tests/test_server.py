"""HTTP contract tests with a fake model: status codes, response shape,
queue limit, error mapping. Needs ffmpeg, not torch.

Run from the repo root: `python -m unittest discover tests`."""

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

import grid  # noqa: E402
import server  # noqa: E402


class FakeModel:
    """Beats every 0.5 s with a snare roll (extra half-beats) at 10-12 s;
    downbeats every 4 beats from beat 1, plus a wrong one on beat 0."""

    def __init__(self, error: Exception | None = None):
        self.error = error

    def __call__(self, pcm, sr):
        if self.error:
            raise self.error
        dur = pcm.size / sr
        beats = [0.25 + 0.5 * i for i in range(int((dur - 0.25) / 0.5) + 1)]
        rolls = [b + 0.25 for b in beats if 10 < b < 12]
        downs = [beats[0]] + beats[1::4]
        return sorted(beats + rolls), downs


def make_wav(seconds: float) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.wav")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
             f"sine=frequency=60:sample_rate=22050:duration={seconds}", path],
            check=True,
        )
        with open(path, "rb") as f:
            return f.read()


WAV_30S = make_wav(30)


class Contract(unittest.TestCase):
    def client(self, model=None):
        patcher = mock.patch.object(server, "_load_model", return_value=model or FakeModel())
        patcher.start()
        self.addCleanup(patcher.stop)
        c = TestClient(server.app, raise_server_exceptions=False)
        c.__enter__()
        self.addCleanup(c.__exit__, None, None, None)
        return c

    def test_healthz(self):
        body = self.client().get("/healthz").json()
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["analyzerVersion"], grid.ANALYZER_VERSION)
        self.assertFalse(body["busy"])

    def test_analyze_audio_returns_a_clean_grid(self):
        r = self.client().post(
            "/analyze-audio", content=WAV_30S,
            headers={"X-Track-Name": "Artist%20-%20Song.mp3"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        for key in ("beats", "downbeatIndices", "downbeatPhase", "bpm", "confidence",
                    "duration", "firstSolidBeat", "analyzerVersion"):
            self.assertIn(key, body)
        self.assertEqual(body["analyzerVersion"], grid.ANALYZER_VERSION)
        self.assertAlmostEqual(body["duration"], 30, delta=0.1)
        self.assertAlmostEqual(body["bpm"], 120, delta=0.1)
        steps = [b - a for a, b in zip(body["beats"], body["beats"][1:])]
        self.assertTrue(all(abs(s - 0.5) < 0.01 for s in steps), "extra beats left in")
        self.assertEqual(body["downbeatPhase"], 1)
        self.assertEqual(body["downbeatIndices"][0] % 4, 1)

    def test_undecodable_is_422_with_reason(self):
        r = self.client().post("/analyze-audio", content=b"not audio at all" * 100)
        self.assertEqual(r.status_code, 422)
        self.assertIn("could not decode audio", r.json()["error"])

    def test_empty_body_is_400(self):
        self.assertEqual(self.client().post("/analyze-audio", content=b"").status_code, 400)

    def test_too_short_is_an_empty_grid(self):
        r = self.client().post("/analyze-audio", content=make_wav(0.5))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["beats"], [])
        self.assertEqual(r.json()["firstSolidBeat"], -1)

    def test_model_failure_is_500(self):
        c = self.client(FakeModel(RuntimeError("bad input shape")))
        r = c.post("/analyze-audio", content=WAV_30S)
        self.assertEqual(r.status_code, 500)
        self.assertIn("bad input shape", r.json()["error"])

    def test_out_of_memory_is_503(self):
        c = self.client(FakeModel(RuntimeError("DefaultCPUAllocator: can't allocate memory")))
        r = c.post("/analyze-audio", content=WAV_30S)
        self.assertEqual(r.status_code, 503)
        self.assertIn("Retry-After", r.headers)
        self.assertIn("allocate", c.get("/healthz").json()["last_error"])

    def test_full_queue_is_503(self):
        c = self.client()
        with mock.patch.dict(server._stats, {"waiting": server.MAX_QUEUE}):
            r = c.post("/analyze-audio", content=WAV_30S)
        self.assertEqual(r.status_code, 503)
        self.assertIn("busy", r.json()["error"])

    def test_grid_bug_falls_back_to_raw_beats(self):
        c = self.client()
        with mock.patch.object(grid, "analyse", side_effect=ValueError("boom")):
            r = c.post("/analyze-audio", content=WAV_30S)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["analyzerVersion"], 1)
        self.assertTrue(r.json()["beats"])

    def test_unexpected_error_is_503_not_500(self):
        c = self.client()
        with mock.patch.object(server, "_analyze_file", side_effect=KeyError("oops")):
            r = c.post("/analyze-audio", content=WAV_30S)
        self.assertEqual(r.status_code, 503)

    def test_analyze_by_path_rejects_traversal(self):
        c = self.client()
        self.assertEqual(c.post("/analyze", json={"path": "../etc/passwd"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
