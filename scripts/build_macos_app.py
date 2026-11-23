#!/usr/bin/env python3
"""Build a locally runnable SwiftUI .app using Apple's command line tools."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import plistlib
import shutil
import subprocess
import tempfile
from pathlib import Path


def clean_bundle_metadata(app: Path) -> None:
    # iCloud can reattach Finder metadata while the generated bundle is signed.
    for key in ("com.apple.FinderInfo", "com.apple.ResourceFork"):
        subprocess.run(["xattr", "-dr", key, str(app)], check=False, capture_output=True)
    # Finish with the root: iCloud may have reattached it during recursive cleanup.
    subprocess.run(["xattr", "-d", "com.apple.FinderInfo", str(app)], check=False, capture_output=True)


def sign_bundle(app: Path) -> None:
    for _ in range(3):
        clean_bundle_metadata(app)
        result = subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", str(app)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            clean_bundle_metadata(app)
            result = subprocess.run(
                ["codesign", "--verify", "--deep", "--strict", str(app)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                return
        if "resource fork, Finder information" not in result.stderr:
            result.check_returncode()
    result.check_returncode()


def verify_bundle(app: Path) -> None:
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True, capture_output=True)


def ensure_app_not_running(app: Path) -> None:
    executable = (app / "Contents" / "MacOS" / "VQAStudio").resolve()
    commands = subprocess.check_output(["ps", "-axo", "comm="], text=True)
    if any(Path(command.strip()).resolve() == executable for command in commands.splitlines() if command.strip()):
        raise RuntimeError(f"Quit the running App before replacing it: {app}")


def bundle_files(app: Path) -> dict[str, str]:
    return {
        str(path.relative_to(app)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in app.rglob("*") if path.is_file()
    }


def package_files(root: Path) -> dict[str, str]:
    package = root / "macos"
    return {
        "Package.swift": hashlib.sha256((package / "Package.swift").read_bytes()).hexdigest(),
        **{f"Sources/{name}": digest for name, digest in bundle_files(package / "Sources").items()},
    }


def snapshot_package(root: Path, destination: Path) -> dict[str, str]:
    expected = package_files(root)
    destination.mkdir()
    shutil.copyfile(root / "macos/Package.swift", destination / "Package.swift")
    shutil.copytree(root / "macos/Sources", destination / "Sources", copy_function=shutil.copyfile)
    if bundle_files(destination) != expected or package_files(root) != expected:
        raise RuntimeError("App sources changed while preparing the build; retry after edits finish")
    return expected


def install_bundle(candidate: Path, destination: Path) -> Path | None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / ".vqa-studio-install.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another App replacement is in progress: {destination}") from error
        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            raise RuntimeError(f"App destination must be a regular directory: {destination}")
        ensure_app_not_running(destination)
        verify_bundle(candidate)
        expected = bundle_files(candidate)
        transaction = Path(tempfile.mkdtemp(prefix=".vqa-studio-install-", dir=destination.parent))
        incoming = transaction / "incoming.app"
        previous = transaction / "previous-bundle"
        try:
            subprocess.run(["ditto", "--norsrc", str(candidate), str(incoming)], check=True)
            clean_bundle_metadata(incoming)
            verify_bundle(incoming)
            if bundle_files(incoming) != expected:
                raise RuntimeError("Copied App files do not match the signed candidate")
            ensure_app_not_running(destination)
            if destination.exists():
                destination.rename(previous)
            try:
                incoming.rename(destination)
                clean_bundle_metadata(destination)
                verify_bundle(destination)
                if bundle_files(destination) != expected:
                    raise RuntimeError("Installed App files do not match the signed candidate")
            except BaseException:
                if destination.exists():
                    destination.rename(transaction / "failed-bundle")
                if previous.exists():
                    previous.rename(destination)
                raise
        except BaseException as error:
            if previous.exists():
                # Never remove the last working bundle if rollback itself fails.
                raise RuntimeError(f"App replacement failed; recover the previous App from {previous}") from error
            shutil.rmtree(transaction)
            raise
        if previous.exists():
            return previous
        transaction.rmdir()
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True, help="Existing project environment's Python executable")
    parser.add_argument("--output", type=Path, help="Destination folder (defaults to the repository root)")
    parser.add_argument("--sdk", type=Path, help="Optional macOS SDK, e.g. a compatible installed 26.5 SDK")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    python = args.python.expanduser().absolute()
    if not python.is_file():
        parser.error(f"Python executable does not exist: {python}")
    output = args.output.expanduser().absolute() if args.output else root
    app = output / "VQA Studio.app"
    ensure_app_not_running(app)
    with tempfile.TemporaryDirectory(prefix="vqa-studio-package-") as workspace:
        package = Path(workspace) / "Package"
        expected_sources = snapshot_package(root, package)
        build_args = ["swift", "build", "--package-path", str(package), "-c", "release"]
        if args.sdk:
            build_args.extend(["--sdk", str(args.sdk)])
        subprocess.run(build_args, check=True)
        binary_dir = subprocess.check_output([*build_args, "--show-bin-path"], text=True).strip()
        if package_files(root) != expected_sources:
            raise RuntimeError("App sources changed during the build; the previous App has been kept")
        candidate = Path(workspace) / "VQA Studio.app"
        assemble_bundle(root, python, Path(binary_dir) / "VQAStudio", candidate)
        sign_bundle(candidate)
        if package_files(root) != expected_sources:
            raise RuntimeError("App sources changed during packaging; the previous App has been kept")
        previous = install_bundle(candidate, app)
    print(f"Built: {app}")
    if previous:
        print(f"Previous App retained: {previous}")
    print("Local build: uses the configured project directory and Python environment. Not notarized for distribution.")


def assemble_bundle(root: Path, python: Path, binary: Path, app: Path) -> None:
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    resources = contents / "Resources"
    resources.mkdir(exist_ok=True)
    executable = contents / "MacOS" / "VQAStudio"
    shutil.copyfile(binary, executable)
    executable.chmod(0o755)
    defaults = {
        "project": str(root), "python": str(python), "config": "configs/default.yaml",
        "checkpoint": "checkpoints/best.pt", "device": "cpu", "offline": True, "topk": 5,
    }
    (resources / "Defaults.json").write_text(json.dumps(defaults, ensure_ascii=False, indent=2) + "\n")
    info = {
        "CFBundleName": "VQA Studio", "CFBundleDisplayName": "VQA Studio",
        "CFBundleExecutable": "VQAStudio", "CFBundleIdentifier": "com.dongtingshuo.vqastudio",
        "CFBundlePackageType": "APPL", "CFBundleShortVersionString": "1.0.0", "CFBundleVersion": "35",
        "LSMinimumSystemVersion": "14.0", "NSHighResolutionCapable": True,
        "NSPrincipalClass": "StudioApplication", "CFBundleIconFile": "FocusQ",
        "NSHumanReadableCopyright": "VQA Studio · Dongting Shuo",
        "CFBundleDevelopmentRegion": "zh-Hans", "CFBundleLocalizations": ["zh-Hans"],
        "CFBundleDocumentTypes": [{"CFBundleTypeName": "图片", "LSItemContentTypes": ["public.image"],
                                   "CFBundleTypeRole": "Viewer", "LSHandlerRank": "Alternate"}],
    }
    with (contents / "Info.plist").open("wb") as handle:
        plistlib.dump(info, handle)
    icon_source = root / "macos" / "Resources" / "AppIcon.png"
    with tempfile.TemporaryDirectory(prefix="vqa-studio-icon-") as icon_workdir:
        subprocess.run(["swift", str(root / "macos" / "make_icon.swift"), icon_workdir, str(icon_source)], check=True)
        subprocess.run(
            ["iconutil", "-c", "icns", str(Path(icon_workdir) / "AppIcon.iconset"),
             "-o", str(resources / "FocusQ.icns")], check=True,
        )


if __name__ == "__main__":
    main()
