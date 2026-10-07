"""Build only World/codex, keeping all generated artifacts outside it."""
import hashlib
import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess

WORLD = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-code-mode-host', action='store_true')
    parser.add_argument('--with-cli', action='store_true', help='Build the local login CLI too')
    args = parser.parse_args()
    source = WORLD / "codex"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    before = subprocess.check_output(["git", "status", "--porcelain"], cwd=source)
    if before:
        raise RuntimeError("Codex source must be clean; refusing to modify or reset it")
    target = WORLD / "var/build/codex" / revision
    target.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    toolchain = WORLD / "var/build/toolchain"
    env.update(CARGO_HOME=str(toolchain / "cargo"), RUSTUP_HOME=str(toolchain / "rustup"),
               CARGO_TARGET_DIR=str(target), CARGO_PROFILE_DEV_DEBUG="0",
               CARGO_NET_GIT_FETCH_WITH_CLI="true", CARGO_HTTP_TIMEOUT="90",
               CARGO_HTTP_MULTIPLEXING="false")
    env["PATH"] = str(toolchain / "cargo/bin") + os.pathsep + env.get("PATH", "")
    native = WORLD / "var/build/native"
    if (native / "usr/bin/pkg-config").exists():
        env["PKG_CONFIG"] = str(native / "usr/bin/pkg-config")
        env["PKG_CONFIG_SYSROOT_DIR"] = str(native)
        env["PKG_CONFIG_LIBDIR"] = str(native / "usr/lib/x86_64-linux-gnu/pkgconfig")
        env["PKG_CONFIG_ALLOW_SYSTEM_CFLAGS"] = "1"
        env["PKG_CONFIG_ALLOW_SYSTEM_LIBS"] = "1"
        env["CFLAGS"] = (env.get("CFLAGS", "") + " -I" +
                         str(native / "usr/include/x86_64-linux-gnu")).strip()
        if list((native / "usr/lib/x86_64-linux-gnu").glob("libclang-*.so*")):
            env["LIBCLANG_PATH"] = str(native / "usr/lib/x86_64-linux-gnu")
    cargo = str(toolchain / "cargo/bin/cargo") if (toolchain / "cargo/bin/cargo").exists() else shutil.which("cargo")
    if not cargo:raise RuntimeError("Install Rust/rustup first; see docs/HANDOFF.md")
    if not (toolchain / "cargo/bin/cargo").exists():
        for key in ("CARGO_HOME","RUSTUP_HOME"):
            if key in os.environ:env[key]=os.environ[key]
            else:env.pop(key,None)
    command = [cargo, "build", "--locked", "-p",
               "codex-app-server", "--bin", "codex-app-server", "-j", "4"]
    if args.with_cli:
        command.extend(['-p','codex-cli','--bin','codex'])
    if args.with_code_mode_host:
        command.extend(['-p', 'codex-code-mode-host', '--bin', 'codex-code-mode-host'])
    with (target / "build.log").open("a") as log:
        log.write("\n=== source build attempt ===\n")
        log.flush()
        result = subprocess.run(command, cwd=source / "codex-rs", env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    after = subprocess.check_output(["git", "status", "--porcelain"], cwd=source)
    if after != before:
        raise RuntimeError("Upstream worktree changed during build; inspect it without resetting")
    result.check_returncode()
    binary = target / "debug/codex-app-server"
    manifest = {"source": str(source), "commit": revision, "binary": str(binary),
                "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "command": command, "locked": True,
                "native_packages": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (WORLD / "var/build").glob("*.deb")}}
    if args.with_code_mode_host:
        helper = target / "debug/codex-code-mode-host"
        manifest["helpers"] = {str(helper): hashlib.sha256(helper.read_bytes()).hexdigest()}
    (target / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(target / "build.json")


if __name__ == "__main__":
    main()
