from __future__ import annotations

import argparse
import ast
import warnings
from pathlib import Path

TOOLKIT_FILES = (
    Path("PythonHelperTools/vqaTools/vqa.py"),
    Path("PythonEvaluationTools/vqaEvaluation/vqaEval.py"),
)

KNOWN_INDENT_FIXES = {
    "         imgToQA =": "        imgToQA =",
    "         for ann in self.dataset['annotations']:": "        for ann in self.dataset['annotations']:",
    "              qqa[ques['question_id']] =": "            qqa[ques['question_id']] =",
    "         # create class members": "        # create class members",
    "         self.qa =": "        self.qa =",
    "         self.imgToQA =": "        self.imgToQA =",
    "                 anns = self.dataset['annotations']": "                anns = self.dataset['annotations']",
    "             anns = anns if len(ansTypes)": "            anns = anns if len(ansTypes)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the official Python 2 VQA toolkit for Python 3.")
    parser.add_argument("--toolkit-root", required=True)
    return parser.parse_args()


def _get_refactoring_tool():
    try:
        from fissix.refactor import RefactoringTool, get_fixers_from_package
    except ImportError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            try:
                from lib2to3.refactor import RefactoringTool, get_fixers_from_package
            except ImportError as exc:
                raise RuntimeError(
                    "Install the 'fissix' dependency to convert the official VQA toolkit "
                    "on this Python version."
                ) from exc
        fixer_package = "lib2to3.fixes"
    else:
        fixer_package = "fissix.fixes"

    return RefactoringTool(get_fixers_from_package(fixer_package))


def _run_2to3(source: str, filename: str) -> str:
    if "print '" not in source and 'print "' not in source:
        return source
    tool = _get_refactoring_tool()
    return str(tool.refactor_string(source, filename))


def convert_source(source: str, filename: str) -> str:
    converted = _run_2to3(source, filename).replace("\t", "    ")
    lines = []
    for line in converted.splitlines(keepends=True):
        for old, new in KNOWN_INDENT_FIXES.items():
            if line.startswith(old):
                line = new + line[len(old) :]
                break
        lines.append(line)
    converted = "".join(lines)
    ast.parse(converted, filename=filename)
    return converted


def prepare_toolkit(toolkit_root: str | Path) -> list[Path]:
    root = Path(toolkit_root)
    changed = []
    for relative_path in TOOLKIT_FILES:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing official toolkit file: {path}")
        source = path.read_text(encoding="utf-8")
        converted = convert_source(source, str(path))
        if converted != source:
            path.write_text(converted, encoding="utf-8")
            changed.append(path)
    return changed


def main() -> None:
    changed = prepare_toolkit(parse_args().toolkit_root)
    if changed:
        for path in changed:
            print(f"prepared: {path}")
    else:
        print("official VQA toolkit is already Python 3 compatible")


if __name__ == "__main__":
    main()
