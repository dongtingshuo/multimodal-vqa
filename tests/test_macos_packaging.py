import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.build_macos_app import sign_bundle

APP = Path("VQA Studio.app")
METADATA_ERROR = "resource fork, Finder information, or similar detritus not allowed"


def result(code=0, error=""):
    return subprocess.CompletedProcess(["codesign"], code, stdout="", stderr=error)


@patch("scripts.build_macos_app.clean_bundle_metadata")
@patch("scripts.build_macos_app.subprocess.run")
def test_signing_verifies_successful_bundle(run, clean):
    run.side_effect = [result(), result()]
    sign_bundle(APP)
    assert "--sign" in run.call_args_list[0].args[0]
    assert "--verify" in run.call_args_list[1].args[0]


@patch("scripts.build_macos_app.clean_bundle_metadata")
@patch("scripts.build_macos_app.subprocess.run")
def test_signing_recovers_from_icloud_metadata_race(run, clean):
    run.side_effect = [result(1, METADATA_ERROR), result(), result()]
    sign_bundle(APP)
    assert run.call_count == 3


@patch("scripts.build_macos_app.clean_bundle_metadata")
@patch("scripts.build_macos_app.subprocess.run")
def test_signing_does_not_hide_unrelated_failure(run, clean):
    run.return_value = result(1, "invalid signing identity")
    with pytest.raises(subprocess.CalledProcessError):
        sign_bundle(APP)
    assert run.call_count == 1


@patch("scripts.build_macos_app.clean_bundle_metadata")
@patch("scripts.build_macos_app.subprocess.run")
def test_signing_reports_persistent_metadata_failure(run, clean):
    run.return_value = result(1, METADATA_ERROR)
    with pytest.raises(subprocess.CalledProcessError):
        sign_bundle(APP)
    assert run.call_count == 3


@pytest.fixture
def installation(tmp_path, monkeypatch):
    import shutil

    import scripts.build_macos_app as packager

    candidate = tmp_path / "signed.app"
    destination = tmp_path / "Output" / "VQA Studio.app"
    for app, content in ((candidate, b"NEW_EXECUTABLE"), (destination, b"OLD_EXECUTABLE")):
        binary = app / "Contents/MacOS/VQAStudio"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(content)
    monkeypatch.setattr(packager, "ensure_app_not_running", lambda app: None)
    monkeypatch.setattr(packager, "verify_bundle", lambda app: None)
    monkeypatch.setattr(packager, "clean_bundle_metadata", lambda app: None)

    def copy_bundle(command, **kwargs):
        assert command[:2] == ["ditto", "--norsrc"]
        shutil.copytree(command[2], command[3])
        return result()

    monkeypatch.setattr(packager.subprocess, "run", copy_bundle)
    return packager, candidate, destination


def test_install_keeps_previous_and_drops_stale_files(installation):
    packager, candidate, destination = installation
    (destination / "stale.txt").write_text("obsolete resource")
    previous = packager.install_bundle(candidate, destination)
    assert packager.bundle_files(destination) == packager.bundle_files(candidate)
    assert (previous / "Contents/MacOS/VQAStudio").read_bytes() == b"OLD_EXECUTABLE"
    assert (previous / "stale.txt").read_text() == "obsolete resource"
    assert previous.suffix != ".app"


def test_candidate_verification_failure_keeps_old_app(installation, monkeypatch):
    packager, candidate, destination = installation
    before = packager.bundle_files(destination)
    monkeypatch.setattr(packager, "verify_bundle", lambda app: (_ for _ in ()).throw(RuntimeError("invalid candidate")))
    with pytest.raises(RuntimeError, match="invalid candidate"):
        packager.install_bundle(candidate, destination)
    assert packager.bundle_files(destination) == before


def test_copy_failure_keeps_old_app(installation, monkeypatch):
    packager, candidate, destination = installation
    before = packager.bundle_files(destination)
    monkeypatch.setattr(packager.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("copy failure")))
    with pytest.raises(OSError, match="copy failure"):
        packager.install_bundle(candidate, destination)
    assert packager.bundle_files(destination) == before


def test_installed_verification_failure_restores_previous(installation, monkeypatch):
    packager, candidate, destination = installation
    before = packager.bundle_files(destination)

    def verify(app):
        if app == destination:
            raise RuntimeError("installed signature failed")

    monkeypatch.setattr(packager, "verify_bundle", verify)
    with pytest.raises(RuntimeError, match="installed signature failed"):
        packager.install_bundle(candidate, destination)
    assert packager.bundle_files(destination) == before


def test_copy_mismatch_is_rejected_before_replacement(installation, monkeypatch):
    packager, candidate, destination = installation
    before = packager.bundle_files(destination)
    copy = packager.subprocess.run

    def corrupt_copy(command, **kwargs):
        response = copy(command, **kwargs)
        (Path(command[3]) / "Contents/MacOS/VQAStudio").write_bytes(b"CORRUPTED")
        return response

    monkeypatch.setattr(packager.subprocess, "run", corrupt_copy)
    with pytest.raises(RuntimeError, match="Copied App files"):
        packager.install_bundle(candidate, destination)
    assert packager.bundle_files(destination) == before


def test_launch_during_staging_prevents_replacement(installation, monkeypatch):
    packager, candidate, destination = installation
    before = packager.bundle_files(destination)
    checks = []

    def check_running(app):
        checks.append(app)
        if len(checks) == 2:
            raise RuntimeError("Quit the running App")

    monkeypatch.setattr(packager, "ensure_app_not_running", check_running)
    with pytest.raises(RuntimeError, match="running App"):
        packager.install_bundle(candidate, destination)
    assert packager.bundle_files(destination) == before


def test_rollback_failure_preserves_recoverable_previous(installation, monkeypatch):
    packager, candidate, destination = installation
    original_rename = Path.rename

    def rename(path, target):
        if path.name == "previous-bundle":
            raise OSError("injected restore failure")
        return original_rename(path, target)

    def verify(app):
        if app == destination:
            raise RuntimeError("installed signature failed")

    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr(packager, "verify_bundle", verify)
    with pytest.raises(RuntimeError, match="recover the previous App from"):
        packager.install_bundle(candidate, destination)
    backups = list(destination.parent.glob(".vqa-studio-install-*/previous-bundle/Contents/MacOS/VQAStudio"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"OLD_EXECUTABLE"


def test_install_rejects_concurrent_replacement(installation):
    import fcntl

    packager, candidate, destination = installation
    before = packager.bundle_files(destination)
    with (destination.parent / ".vqa-studio-install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another App replacement"):
            packager.install_bundle(candidate, destination)
    assert packager.bundle_files(destination) == before


def test_main_sign_failure_leaves_previous_bundle_untouched(tmp_path, monkeypatch):
    import scripts.build_macos_app as packager

    project = tmp_path / "Project"
    (project / "scripts").mkdir(parents=True)
    (project / "macos/Sources/VQAStudio").mkdir(parents=True)
    (project / "macos/Package.swift").write_text("// fixture manifest")
    (project / "macos/Sources/VQAStudio/Main.swift").write_text("// fixture source")
    destination = project / "VQA Studio.app"
    (destination / "Contents").mkdir(parents=True)
    (destination / "Contents/old.txt").write_text("previous working app")
    before = packager.bundle_files(destination)
    monkeypatch.setattr(packager, "__file__", str(project / "scripts/build_macos_app.py"))
    monkeypatch.setattr("sys.argv", ["build", "--python", str(Path(__file__))])
    monkeypatch.setattr(packager, "ensure_app_not_running", lambda app: None)
    monkeypatch.setattr(packager.subprocess, "run", lambda *a, **k: result())
    monkeypatch.setattr(packager.subprocess, "check_output", lambda *a, **k: str(tmp_path))

    def assemble(root, python, binary, app):
        app.mkdir()
        (app / "new.txt").write_text("new candidate")

    monkeypatch.setattr(packager, "assemble_bundle", assemble)
    monkeypatch.setattr(packager, "sign_bundle", lambda app: (_ for _ in ()).throw(RuntimeError("injected signing failure")))
    with pytest.raises(RuntimeError, match="injected signing failure"):
        packager.main()
    assert packager.bundle_files(destination) == before


def test_running_app_detection_handles_spaces(tmp_path, monkeypatch):
    import scripts.build_macos_app as packager

    app = tmp_path / "Project With Spaces" / "VQA Studio.app"
    executable = app / "Contents/MacOS/VQAStudio"
    monkeypatch.setattr(packager.subprocess, "check_output", lambda *a, **k: f"/usr/bin/other\n{executable}\n")
    with pytest.raises(RuntimeError, match="Quit the running App"):
        packager.ensure_app_not_running(app)
    packager.ensure_app_not_running(tmp_path / "Different.app")


def test_source_snapshot_excludes_cloud_build_database(tmp_path):
    import scripts.build_macos_app as packager

    project = tmp_path / "Project"
    source = project / "macos/Sources/VQAStudio/Main.swift"
    source.parent.mkdir(parents=True)
    source.write_text("// stable source")
    (project / "macos/Package.swift").write_text("// manifest")
    (project / "macos/.build").mkdir()
    (project / "macos/.build/build.db").write_text("unusable cloud database")
    snapshot = tmp_path / "Snapshot"
    expected = packager.snapshot_package(project, snapshot)
    assert packager.bundle_files(snapshot) == expected
    assert not (snapshot / ".build").exists()
    source.write_text("// later edit")
    assert (snapshot / "Sources/VQAStudio/Main.swift").read_text() == "// stable source"
    assert packager.package_files(project) != expected


def test_source_snapshot_rejects_change_during_copy(tmp_path, monkeypatch):
    import scripts.build_macos_app as packager

    project = tmp_path / "Project"
    source = project / "macos/Sources/VQAStudio/Main.swift"
    source.parent.mkdir(parents=True)
    source.write_text("// original")
    (project / "macos/Package.swift").write_text("// manifest")
    copy = packager.shutil.copyfile

    def edit_during_copy(path, destination, **kwargs):
        if Path(path) == source:
            source.write_text("// changed during copy")
        return copy(path, destination, **kwargs)

    monkeypatch.setattr(packager.shutil, "copyfile", edit_during_copy)
    with pytest.raises(RuntimeError, match="sources changed"):
        packager.snapshot_package(project, tmp_path / "Snapshot")


def test_source_snapshot_ignores_metadata_only_changes(tmp_path):
    import os

    import scripts.build_macos_app as packager

    project = tmp_path / "Project"
    source = project / "macos/Sources/VQAStudio/Main.swift"
    source.parent.mkdir(parents=True)
    source.write_text("// stable source")
    (project / "macos/Package.swift").write_text("// manifest")
    expected = packager.snapshot_package(project, tmp_path / "Snapshot")
    os.utime(source, (1, 1))
    assert packager.package_files(project) == expected
