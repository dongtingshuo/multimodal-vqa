from __future__ import annotations

import ast

import pytest

from scripts.prepare_official_vqa_toolkit import convert_source


def test_convert_source_ports_python2_print_syntax() -> None:
    converted = convert_source("def demo():\n\tprint 'ready'\n", "demo.py")
    ast.parse(converted)
    assert "print('ready')" in converted
    assert convert_source(converted, "demo.py") == converted


def test_convert_source_repairs_known_official_indentation() -> None:
    source = (
        "def createIndex(self):\n"
        "        print('creating index...')\n"
        "         imgToQA = {}\n"
        "         self.qa = imgToQA\n"
    )
    converted = convert_source(source, "vqa.py")
    ast.parse(converted)
    assert "        imgToQA = {}" in converted
    assert "        self.qa = imgToQA" in converted


def test_conversion_reports_missing_refactoring_dependency(monkeypatch) -> None:
    original_import = __import__

    def import_without_refactor(name, *args, **kwargs):
        if name == "fissix.refactor" or name.startswith("lib2to3"):
            raise ImportError("simulated missing refactoring dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_without_refactor)
    with pytest.raises(RuntimeError, match="fissix"):
        convert_source("print 'ready'\\n", "demo.py")
