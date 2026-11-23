"""Private JSON-lines transport for the native macOS client (no web server)."""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path


class DesktopPredictor:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "auto", offline: bool = True):
        if offline:
            os.environ["VQA_HF_LOCAL_ONLY"] = "1"
            os.environ["VQA_BEIT3_LOCAL_ONLY"] = "1"
        else:
            os.environ.pop("VQA_HF_LOCAL_ONLY", None)
            os.environ.pop("VQA_BEIT3_LOCAL_ONLY", None)

        from .answers import AnswerVocab
        from .config import apply_runtime_overrides, load_config, resolve_checkpoint_config, resolve_device
        from .engine import load_checkpoint
        from .inputs import build_input_pipeline
        from .model import build_model

        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("未找到模型文件，请在设置中选择已有的 .pt 权重文件。")
        config = apply_runtime_overrides(load_config(config_path), device=device)
        self.device = resolve_device(config["device"], allow_fallback=True)
        checkpoint = load_checkpoint(path, "cpu")
        self.config = resolve_checkpoint_config(config, checkpoint)
        self.vocab = AnswerVocab(
            checkpoint.get("idx_to_answer")
            or AnswerVocab.load(self.config["data"]["answer_vocab_path"]).idx_to_answer,
            normalization=str(checkpoint.get("answer_vocab_normalization", "legacy")),
        )
        tokenizer = path.parent / "beit3.spm"
        self.pipeline = build_input_pipeline(
            self.config["model"], self.config["data"],
            tokenizer_path=tokenizer if tokenizer.is_file() else None,
        )
        self.model = build_model(
            self.config["model"], answer_vocab_size=len(self.vocab), load_pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()
        self.info = {
            "model": str(self.config["model"].get("name", "cross_attention")),
            "device": str(self.device), "vocabulary": len(self.vocab),
            "checkpoint": path.name,
        }
        del checkpoint

    def predict(self, request: dict) -> dict:
        from .inference import predict

        question = request.get("question", "")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("请输入一个英文问题。")
        topk = request.get("topk", 5)
        if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= 10:
            raise ValueError("候选答案数量必须为 1–10。")
        image = Path(request["image"]).expanduser()
        if not image.is_file():
            raise FileNotFoundError("图片已移动或被删除，请重新导入。")
        start = time.monotonic()
        results = predict(
            model=self.model, input_pipeline=self.pipeline, answer_vocab=self.vocab,
            image_path=image, question=question.strip(), device=self.device,
            image_size=self.config["data"]["image_size"],
            max_question_length=self.config["data"]["max_question_length"], topk=topk,
        )
        return {
            "answers": [{"answer": answer, "score": score} for answer, score in results],
            "elapsed": time.monotonic() - start, **self.info,
        }


def serve(input_stream=None, output_stream=None, factory=DesktopPredictor) -> None:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    predictor = None
    signature = None
    for line in input_stream:
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("请求必须为 JSON 对象。")
            request_id = request.get("id")
            action = request.get("action")
            if action not in {"load", "predict"}:
                raise ValueError("不支持的操作。")
            settings = request["settings"]
            new_signature = json.dumps(settings, sort_keys=True)
            # Libraries may print status messages. stdout belongs exclusively to the protocol.
            with contextlib.redirect_stdout(sys.stderr):
                if predictor is None or signature != new_signature:
                    predictor = None
                    signature = None
                    predictor = factory(**settings)
                    signature = new_signature
                result = predictor.info if action == "load" else predictor.predict(request)
            response = {"id": request_id, "ok": True, **result}
            encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
        except Exception as exc:
            response = {"id": request_id, "ok": False, "error": str(exc)}
            encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
        try:
            output_stream.write(encoded + "\n")
            output_stream.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The desktop client may close its pipe during normal quit/cancellation.
            # Prevent Python's final stdout flush from raising again (exit status 120).
            if output_stream is sys.stdout:
                descriptor = os.open(os.devnull, os.O_WRONLY)
                try:
                    os.dup2(descriptor, output_stream.fileno())
                finally:
                    os.close(descriptor)
            return


if __name__ == "__main__":
    serve()
