"""CPU-only package/build inspection; does not launch Isaac or claim GPU compatibility."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--world', type=Path, help='Optional read-only World mount for isolated coding worker CPU check')
args = parser.parse_args()

result = {"python": sys.version, "platform": platform.platform(), "gpu_executed": False,
          "packages": {}, "version_files": {}}
for package in ("numpy", "torch", "torchvision", "isaaclab", "isaaclab-assets", "isaaclab-tasks", "warp-lang"):
    try:
        result["packages"][package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        result["packages"][package] = None
for path in ("/isaac-sim/VERSION", "/isaac-sim/kit/VERSION", "/opt/isaaclab21/VERSION"):
    if Path(path).is_file():
        result["version_files"][path] = Path(path).read_text().strip()
import torch
result["torch_cuda_build"] = torch.version.cuda
flag_reader = getattr(torch._C, "_cuda_getArchFlags", None)
result["torch_compiled_cuda_arch_flags"] = flag_reader() if flag_reader else None
if args.world:
    sys.path.insert(0, str(args.world))
    from environment.benchmarks.humanoid_soccer.coding import ControllerProgram
    program = ControllerProgram('def control(obs, memory):\n    return [0.0] * 21\n')
    try:
        response = program.request({'obs': {'native_actor_history': [0.] * 405}, 'memory': {}})
        result['isolated_coding_worker'] = {'response': response, 'gpu_executed': False}
        if response.get('action') != [0.] * 21:
            raise RuntimeError('Unexpected isolated worker action response')
    finally:
        program.close()
result["interpretation"] = "Compile flags are build metadata only. No CUDA kernel, PhysX or renderer has been tested."
print(json.dumps(result, indent=2))
