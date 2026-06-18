from __future__ import annotations

import argparse
import os
import socket
import tempfile
from pathlib import Path

cache_root = Path("outputs/cache").resolve()
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import gradio as gr
from PIL import Image

from vqa_project.answers import AnswerVocab
from vqa_project.config import load_config, resolve_device
from vqa_project.engine import load_checkpoint
from vqa_project.hf import load_tokenizer
from vqa_project.inference import predict
from vqa_project.model import VQAModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Gradio VQA demo.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--server-port", type=int, default=8877)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--inbrowser", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Only use locally cached Hugging Face model files.")
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def find_available_port(preferred_port: int, server_name: str, attempts: int = 50) -> int:
    for port in range(preferred_port, preferred_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((server_name, port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found in range {preferred_port}-{preferred_port + attempts - 1}")


def build_predictor(config_path: str, checkpoint_path: str):
    config = load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    device = resolve_device(config["device"], allow_fallback=True)
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        message = (
            f"未找到模型文件：{checkpoint_file}。请先运行 "
            "`python train.py --config configs/default.yaml` 训练并保存 checkpoint。"
        )

        def missing_checkpoint(_image, _question, _topk):
            return message

        return missing_checkpoint

    try:
        checkpoint = load_checkpoint(checkpoint_file, device)
        answer_vocab = AnswerVocab(
            checkpoint.get("idx_to_answer") or AnswerVocab.load(data_cfg["answer_vocab_path"]).idx_to_answer
        )
        tokenizer = load_tokenizer(model_cfg["text_model_name"])
        model = VQAModel(answer_vocab_size=len(answer_vocab), **model_cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
    except Exception as exc:
        message = f"模型加载失败：{exc}"

        def failed_model_load(_image, _question, _topk):
            return message

        return failed_model_load

    def run(image: Image.Image | None, question: str, topk: int):
        if image is None:
            return "请先上传一张图片。"
        if not question or not question.strip():
            return "请输入一个英文问题。"
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.convert("RGB").save(tmp.name)
            temp_path = tmp.name
        results = predict(
            model=model,
            tokenizer=tokenizer,
            answer_vocab=answer_vocab,
            image_path=temp_path,
            question=question,
            device=device,
            image_size=data_cfg["image_size"],
            max_question_length=data_cfg["max_question_length"],
            topk=int(topk),
        )
        Path(temp_path).unlink(missing_ok=True)
        return "\n".join(f"{answer}\t{probability:.4f}" for answer, probability in results)

    return run


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ["VQA_HF_LOCAL_ONLY"] = "1"
    predictor = build_predictor(args.config, args.checkpoint)
    server_port = find_available_port(args.server_port, args.server_name)
    if server_port != args.server_port:
        print(f"Port {args.server_port} is busy; using {server_port} instead.")
    with gr.Blocks(title="VQA Demo") as demo:
        gr.Markdown("# 基于多模态融合的视觉问答（VQA）")
        with gr.Row():
            image = gr.Image(type="pil", label="Image")
            with gr.Column():
                question = gr.Textbox(label="Question", value="What is in the image?")
                topk = gr.Slider(1, 10, value=5, step=1, label="Top-k")
                submit = gr.Button("Predict", variant="primary")
        output = gr.Textbox(label="Predictions", lines=8)
        submit.click(predictor, inputs=[image, question, topk], outputs=output)
    demo.launch(
        server_name=args.server_name,
        server_port=server_port,
        inbrowser=args.inbrowser,
        share=args.share,
        show_api=False,
    )


if __name__ == "__main__":
    main()
