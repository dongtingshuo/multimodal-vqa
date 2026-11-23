#!/usr/bin/env python3
"""Run isolated native desktop regression checks without XCTest or model weights."""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = root / "macos/Sources/VQAStudio"
    names = [
        "AppSettings", "ArchivedImage", "BackendClient", "BackendError", "RequestDeadline", "ConversationExporter", "ImageImport",
        "ImageSession", "ImageZoom", "LibraryQuery", "LibrarySortOrder", "LibraryState", "LibraryStorage",
        "PendingQuestion", "Prediction", "QuestionDrafts", "QuestionRecord", "StudioImageCache",
        "StudioNotice", "StudioStore", "WorkerResponse",
        "ImageFileStamp", "DecodedImage", "ImageCodec", "ImageDecodingService", "ImagePreview",
        "CachedThumbnail", "ConversationScrollTarget", "CopyFeedback", "DraftReplacement",
        "LibraryConflictError", "PredictionImage", "ImageSnapshotError",
        "StudioAppDelegate", "StudioFilePanel", "StudioErrorEvent", "StudioErrorPresentation",
        "ImageImportRequest",
        "WorkspaceFocus",
        "ImageImportClient", "StudioHelperProcess", "StudioHelperCleanup", "ImageImportWorker",
        "SettingsValidationClient", "SettingsValidationWorker",
        "ImageReadReceipt", "ImageReadError", "ImageWorkLimiter",
    ]
    with tempfile.TemporaryDirectory(prefix="vqa-desktop-checks-") as folder:
        temporary = Path(folder)
        original_sources = sources
        sources = temporary / "Sources"
        shutil.copytree(original_sources, sources, copy_function=shutil.copyfile)
        if any(path.read_bytes() != (original_sources / path.name).read_bytes() for path in sources.glob("*.swift")):
            raise RuntimeError("App sources changed while preparing native checks")
        worker = temporary / "worker/vqa_project"
        worker.mkdir(parents=True)
        (worker / "__init__.py").write_text("")
        for name in ("unused.yaml", "unused.pt", "held-load.yaml"):
            (worker.parent / name).write_text("PROTOCOL_TEST_ONLY; not model data")
        (worker / "desktop_worker.py").write_text('''import json, os, pathlib, signal, sys, time
root = pathlib.Path(__file__).resolve().parents[1]
(root / 'startup-info.json').write_text(json.dumps({'cwd': os.getcwd(), 'import_root': sys.path[0], 'relative_config': pathlib.Path('unused.yaml').read_text()}))
(root / 'worker.pid').write_text(str(os.getpid()))
for line in sys.stdin:
    request = json.loads(line)
    (root / 'last-request.json').write_text(json.dumps(request))
    action = request['action']
    question = request.get('question')
    operation = question if action == 'predict' else action
    if action == 'load' and request['settings']['config_path'].endswith('held-load.yaml'):
        (root / 'startup-entered').write_text('PROTOCOL_TEST_ONLY')
        while (root / 'hold-startup').exists(): time.sleep(0.01)
    if operation in {'wait', 'stubborn'}:
        if operation == 'stubborn': signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (root / ('stubborn.pid' if operation == 'stubborn' else 'waiting.pid')).write_text(str(os.getpid()))
        time.sleep(30)
    if operation == 'abort': sys.exit(13)
    if action == 'malformed':
        print('not json', flush=True)
        continue
    response = {'id': request['id'], 'ok': True, 'model': 'PROTOCOL_TEST_ONLY', 'device': 'cpu'}
    if action in {'idle-stubborn', 'malformed-stubborn', 'closed-output-stubborn'} or (action == 'load' and request['settings']['config_path'].endswith('idle-load.yaml')):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (root / 'retiring.pid').write_text(str(os.getpid()))
        if action == 'malformed-stubborn': print('not json', flush=True)
        elif action == 'closed-output-stubborn': os.close(sys.stdout.fileno())
        else: print(json.dumps(response), flush=True)
        time.sleep(30)
        break
    if action == 'partial':
        sys.stdout.write(json.dumps(response))
        sys.stdout.flush()
        break
    if action == 'mismatch': response['id'] = 'wrong-request-id'
    if action == 'rejected': response.update(ok=False, error='explicit protocol test rejection')
    if action == 'predict':
        response.update(answers=[{'answer': 'PROTOCOL_TEST_ONLY', 'score': 0.5}], elapsed=0.1)
        if question == 'score-outside-range': response['answers'][0]['score'] = 1.5
        if question == 'duplicate-answers': response['answers'] *= 2
        if question == 'negative-elapsed': response['elapsed'] = -1
        if question == 'topk-check': response['answers'] = [{'answer': f'PROTOCOL_TEST_ONLY_{i}', 'score': 0.5} for i in range(request['topk'])]
    print(json.dumps(response), flush=True)
''')
        compiler = ["swiftc", "-sdk", str(args.sdk), "-target", f"{platform.machine()}-apple-macos14.0"]
        draft_checks = temporary / "QuestionDraftsChecks"
        subprocess.run([
            *compiler, str(sources / "QuestionDrafts.swift"),
            str(root / "macos/Tests/QuestionDraftsChecks.swift"), "-o", str(draft_checks),
        ], check=True)
        subprocess.run([str(draft_checks)], check=True, timeout=15)
        desktop_checks = temporary / "DesktopChecks"
        subprocess.run([
            *compiler, *(str(sources / f"{name}.swift") for name in names),
            str(root / "macos/Tests/DesktopChecks.swift"), "-o", str(desktop_checks),
        ], check=True)
        subprocess.run([str(desktop_checks), str(temporary), sys.executable], check=True, timeout=40)
        reliability_checks = temporary / "ReliabilityChecks"
        subprocess.run([
            *compiler, *(str(sources / f"{name}.swift") for name in names),
            str(root / "macos/Tests/ReliabilityChecks.swift"), "-o", str(reliability_checks),
        ], check=True)
        subprocess.run([str(reliability_checks), str(temporary), sys.executable], check=True, timeout=45)
        image_checks = temporary / "ImageWorkflowChecks"
        subprocess.run([
            *compiler, *(str(sources / f"{name}.swift") for name in names),
            str(root / "macos/Tests/ImportTestGate.swift"),
            str(root / "macos/Tests/ImageWorkflowChecks.swift"), "-o", str(image_checks),
        ], check=True)
        subprocess.run([str(image_checks), str(temporary)], check=True, timeout=30)
        import_checks = temporary / "ImageImportChecks"
        subprocess.run([
            *compiler, *(str(sources / f"{name}.swift") for name in names),
            str(root / "macos/Tests/ImageImportChecks.swift"), "-o", str(import_checks),
        ], check=True)
        subprocess.run([str(import_checks), str(temporary)], check=True, timeout=30)
        snapshot_checks = temporary / "SnapshotWorkflowChecks"
        subprocess.run([
            *compiler, *(str(sources / f"{name}.swift") for name in names),
            str(root / "macos/Tests/SnapshotWorkflowChecks.swift"), "-o", str(snapshot_checks),
        ], check=True)
        subprocess.run([str(snapshot_checks), str(temporary), sys.executable], check=True, timeout=30)
        lifecycle_checks = temporary / "BackendLifecycleChecks"
        subprocess.run([
            *compiler, *(str(sources / f"{name}.swift") for name in names),
            str(root / "macos/Tests/BackendLifecycleChecks.swift"), "-o", str(lifecycle_checks),
        ], check=True)
        subprocess.run([str(lifecycle_checks), str(temporary), sys.executable], check=True, timeout=30)
        settings_checks = temporary / "SettingsValidationChecks"
        subprocess.run([
            *compiler, *(str(sources / f"{name}.swift") for name in names),
            str(root / "macos/Tests/SettingsValidationChecks.swift"), "-o", str(settings_checks),
        ], check=True)
        subprocess.run([str(settings_checks), str(temporary), sys.executable], check=True, timeout=20)


if __name__ == "__main__":
    main()
