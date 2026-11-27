"""Host-only GPU discovery and lifecycle ownership for one four-lane Docker batch."""
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

BASE = Path("/dockerdata/robotworld-r580")
TRIAL = BASE / "docker-trial-20260925"
DOCKER = TRIAL / "bin/docker"
STATE = Path("/root/robotworld-behavior-private/docker-parallel-runtime")
HOST = f"unix://{STATE}/run/docker.sock"
IMAGE = "robotworld-behavior-python-compiler:ubuntu24.04-py311"


def discover_gpus(indices, query=None, drm=Path("/sys/class/drm")):
    if len(set(indices)) != len(indices):
        raise ValueError("Each parallel lane requires a distinct GPU")
    if query is None:
        query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,driver_version,memory.used",
                                         "--format=csv,noheader,nounits"], text=True)
    rows = {int(r[0]): r for r in csv.reader(query.splitlines(), skipinitialspace=True)}
    result = []
    for index in indices:
        row = rows[index]
        _, uuid, pci, driver, memory = row
        if driver != "580.105.08" or int(memory) != 0:
            raise RuntimeError(f"GPU {index} is occupied or has an unexpected driver")
        pci = pci.lower()[-12:]
        devices = {}
        for pattern, name in (("renderD*", "render"), ("card[0-9]*", "card")):
            matches = [p for p in drm.glob(pattern) if (p / "device").resolve().name.lower() == pci
                       and "-" not in p.name]
            if len(matches) != 1:
                raise RuntimeError(f"GPU {index}: expected exactly one {name} for {pci}")
            devices[name] = "/dev/dri/" + matches[0].name
        result.append(dict(index=index, uuid=uuid, pci=pci, driver=driver, **devices))
    return result


def docker(*args, **kwargs):
    return subprocess.run([str(DOCKER), "--host", HOST, *args], check=True,
                          text=True, capture_output=True, **kwargs)


class BatchDaemon:
    """The supervisor owns daemon cleanup; lane launchers never stop the daemon."""
    def __init__(self, batch):
        self.batch = Path(batch)
        self.process = None
        self.lock = self.log = None

    def __enter__(self):
        self.lock = (BASE / "docker-trial.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if any((root / relative).exists() for root in (TRIAL, STATE)
                   for relative in ("run/docker.sock", "run/dockerd.pid")):
                raise RuntimeError("Existing Docker daemon left untouched; wait for its owner")
            if not (STATE / "data/image/vfs").is_dir():
                raise RuntimeError("Prepare the separate offline Docker data copy before parallel startup")
            space = os.statvfs(STATE)
            if space.f_favail < 250000 or space.f_bavail * space.f_frsize < 16 * 1024**3:
                raise RuntimeError("Parallel Docker storage needs 250000 free inodes and 16 GiB")
            for directory in ("run", "exec"):
                (STATE / directory).mkdir(exist_ok=True)
            env = os.environ.copy()
            env.update(PATH=f"{TRIAL}/bin:{TRIAL}/toolkit/root/usr/bin:/usr/bin:/bin",
                       LD_LIBRARY_PATH=f"{TRIAL}/toolkit/root/usr/lib64:/lib64",
                       NVIDIA_CONTAINER_RUNTIME_CONFIG=f"{TRIAL}/nvidia-container-runtime/config.toml")
            cmd = [str(TRIAL / "bin/dockerd"), "--host", HOST, "--pidfile", str(STATE / "run/dockerd.pid"),
                   "--data-root", str(STATE / "data"), "--exec-root", str(STATE / "exec"),
                   "--config-file", str(TRIAL / "daemon.json"), "--storage-driver", "vfs",
                   "--bridge", "none", "--iptables=false", "--ip6tables=false", "--ip-forward=false",
                   "--ip-masq=false", "--log-level", "warn"]
            self.log = (self.batch / "daemon.log").open("w")
            self.process = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            for _ in range(150):
                if self.process.poll() is not None:
                    raise RuntimeError("Managed daemon exited during startup")
                try:
                    docker("info", timeout=2)
                    break
                except (subprocess.SubprocessError, OSError):
                    time.sleep(.2)
            else:
                raise TimeoutError("Managed daemon not ready")
            image = json.loads(docker("image", "inspect", IMAGE).stdout)[0]["Id"]
            marker = dict(batch=str(self.batch), daemon_pid=self.process.pid,
                          supervisor_pid=os.getpid(), docker_host=HOST, image=image,
                          data_root=str(STATE / "data"))
            (self.batch / "daemon-owner.json").write_text(json.dumps(marker, indent=2) + "\n")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        try:
            if self.process is not None and self.process.poll() is None:
                # Only containers bearing this exact batch label may be removed here.
                owned = docker("ps", "-aq", "--filter", f"label=robotworld.batch={self.batch}").stdout.split()
                for container in owned:
                    docker("stop", "--timeout", "15", container, timeout=30)
                    docker("rm", container, timeout=15)
                others = docker("ps", "-q").stdout.strip()
                if others:
                    raise RuntimeError("Foreign running containers found; managed daemon left untouched")
                self.process.send_signal(signal.SIGINT)
                self.process.wait(timeout=45)
        finally:
            if self.log:
                self.log.close()
            if self.lock:
                self.lock.close()


def assignment(batch, gpu):
    batch = Path(batch).resolve(strict=True)
    private = Path("/root/robotworld-behavior-private")
    if batch.parent != private or not batch.name.startswith("first-four-parallel-"):
        raise ValueError("Invalid managed batch path")
    owner = json.loads((batch / "daemon-owner.json").read_text())
    pid = int((STATE / "run/dockerd.pid").read_text())
    if owner["batch"] != str(batch) or owner["daemon_pid"] != pid or owner["docker_host"] != HOST:
        raise ValueError("Docker daemon ownership mismatch")
    os.kill(owner["supervisor_pid"], 0)
    lanes = json.loads((batch / "gpu-assignments.json").read_text())
    matches = [lane for lane in lanes if lane["index"] == gpu]
    if len(matches) != 1:
        raise ValueError("GPU not assigned to this batch")
    return matches[0]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    lane = assignment(args.batch, args.gpu)
    for key in ("uuid", "render", "card", "pci", "cpus"):
        print(lane[key])
