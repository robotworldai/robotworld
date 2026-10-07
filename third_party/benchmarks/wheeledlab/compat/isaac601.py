"""External runtime aliases and generated-output routing; no upstream edits."""
from pathlib import Path
import importlib.util

def install(output):
    world=Path(__file__).resolve().parents[4]
    import carb
    asset_root=world/'third_party/benchmarks/wheeledlab/assets'
    carb.settings.get_settings().set('/persistent/isaac/asset_root/cloud',str(asset_root))
    path=world/'third_party/benchmarks/robolab/compat/isaac601.py'
    spec=importlib.util.spec_from_file_location('world_wheeledlab_lab22_compat',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.install()
    from isaaclab.sim import GroundPlaneCfg
    GroundPlaneCfg.usd_path=str(asset_root/'Isaac/Environments/Grid/default_environment.usd')
    # Upstream visual config writes a generated map at module import. Route those
    # outputs to the run directory while retaining original robot/terrain assets.
    import wheeledlab_assets
    original=Path(wheeledlab_assets.WHEELEDLAB_ASSETS_DATA_DIR)
    generated=Path(output)/'generated_assets';generated.mkdir(exist_ok=True)
    for name in ['Robots','Terrains']:
        (generated/name).symlink_to(original/name,target_is_directory=True)
    (generated/'rgb_maps').mkdir()
    wheeledlab_assets.WHEELEDLAB_ASSETS_DATA_DIR=str(generated)
