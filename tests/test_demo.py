from __future__ import annotations

import argparse

import demo


def test_demo_uses_gradio_6_launch_options(monkeypatch) -> None:
    launch_options = {}
    args = argparse.Namespace(
        config="configs/default.yaml",
        checkpoint="checkpoints/best.pt",
        server_port=8877,
        server_name="127.0.0.1",
        inbrowser=False,
        offline=False,
        share=False,
        device=None,
    )

    monkeypatch.setattr(demo, "parse_args", lambda: args)
    monkeypatch.setattr(demo, "build_predictor", lambda *_args: (lambda *_inputs: "answer", "loaded"))
    monkeypatch.setattr(demo, "find_available_port", lambda port, _server_name: port)
    monkeypatch.setattr(demo.gr.Blocks, "launch", lambda _self, **kwargs: launch_options.update(kwargs))

    demo.main()

    assert launch_options["footer_links"] == ["gradio", "settings"]
    assert "show_api" not in launch_options
