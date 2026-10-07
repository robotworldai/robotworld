"""Independent precision-task configuration, reusing the upstream vehicle/actions."""
def build(spec, output):
    from pathlib import Path
    from .environment import build as base_build
    from .precision_geometry import generate
    cfg=base_build(spec,output)
    cfg.scene.course.spawn.usd_path=str(generate(spec,Path(output)/'generated_assets/precision-course.usda'))
    # This switch exists upstream; only this new task family enables it.
    cfg.actions.throttle_steer.no_reverse=False
    cfg.viewer.eye=[7,-8,8];cfg.viewer.lookat=[0,0,0]
    return cfg
