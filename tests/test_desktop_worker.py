from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vqa_project.desktop_worker import serve


class StubPredictor:
    loads = 0

    def __init__(self, **settings):
        type(self).loads += 1
        print("third party model diagnostic")
        if settings.get("checkpoint_path") == "missing":
            raise FileNotFoundError("missing checkpoint")
        self.info = {"model": "test-double", "device": "cpu"}

    def predict(self, request):
        if not request.get("question"):
            raise ValueError("question required")
        return {"answers": [{"answer": "unit-test-only", "score": 0.25}], "elapsed": 0.1}


def run_protocol(requests):
    output = io.StringIO()
    serve(io.StringIO("\n".join(json.dumps(r) for r in requests)), output, StubPredictor)
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_reuses_loaded_model_and_keeps_diagnostics_out_of_protocol(capsys):
    StubPredictor.loads = 0
    settings = {"checkpoint_path": "test"}
    responses = run_protocol([
        {"id": "a", "action": "load", "settings": settings},
        {"id": "b", "action": "predict", "settings": settings, "question": "test"},
        {"id": "c", "action": "load", "settings": {"checkpoint_path": "other"}},
    ])
    assert [r["id"] for r in responses] == ["a", "b", "c"]
    assert all(r["ok"] for r in responses)
    assert StubPredictor.loads == 2
    assert "third party model diagnostic" in capsys.readouterr().err


def test_errors_do_not_poison_subsequent_requests():
    responses = run_protocol([
        {"id": "a", "action": "load", "settings": {"checkpoint_path": "missing"}},
        {"id": "b", "action": "predict", "settings": {}, "question": ""},
        {"id": "c", "action": "load", "settings": {}},
        {"id": "d", "action": "unsupported", "settings": {}},
    ])
    assert responses[0]["error"] == "missing checkpoint"
    assert responses[1]["error"] == "question required"
    assert responses[2]["ok"]
    assert not responses[3]["ok"]


def test_malformed_input_returns_error_and_worker_continues():
    output = io.StringIO()
    serve(io.StringIO('not-json\n[]\n{"id":"ok","action":"load","settings":{}}\n'), output, StubPredictor)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [r["ok"] for r in responses] == [False, False, True]
    assert responses[-1]["id"] == "ok"


def test_unserializable_prediction_returns_error_without_ending_worker():
    class InvalidResult(StubPredictor):
        def predict(self, request):
            return {"answers": [{"answer": "invalid-test-only", "score": float("nan")}]}

    output = io.StringIO()
    requests = [
        {"id": "invalid", "action": "predict", "settings": {}, "question": "test"},
        {"id": "recovery", "action": "load", "settings": {}},
    ]
    serve(io.StringIO("\n".join(json.dumps(r) for r in requests)), output, InvalidResult)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert responses[0]["id"] == "invalid" and not responses[0]["ok"]
    assert responses[1]["id"] == "recovery" and responses[1]["ok"]


@pytest.mark.parametrize("stage", ["write", "flush"])
def test_closed_desktop_connection_stops_without_processing_more_requests(stage):
    class DisconnectedOutput(io.StringIO):
        def write(self, text):
            if stage == "write":
                raise BrokenPipeError("desktop closed")
            return super().write(text)

        def flush(self):
            if stage == "flush":
                raise ConnectionResetError("desktop closed")
            return super().flush()

    StubPredictor.loads = 0
    requests = [
        {"id": "first", "action": "load", "settings": {}},
        {"id": "never-started", "action": "load", "settings": {"checkpoint_path": "other"}},
    ]
    serve(io.StringIO("\n".join(json.dumps(r) for r in requests)), DisconnectedOutput(), StubPredictor)
    assert StubPredictor.loads == 1


def test_real_closed_stdout_exits_cleanly_without_finalizer_traceback():
    program = "from types import SimpleNamespace; from vqa_project.desktop_worker import serve; serve(factory=lambda **_: SimpleNamespace(info={}))"
    process = subprocess.Popen(
        [sys.executable, "-c", program], cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        process.stdout.close()
        process.stdin.write(json.dumps({"id": "closed", "action": "load", "settings": {}}) + "\n")
        process.stdin.close()
        assert process.wait(timeout=5) == 0
        assert "BrokenPipe" not in process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stderr.close()
