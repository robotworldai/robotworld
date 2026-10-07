CASES={
    'mushr-drift': {'id':'Isaac-MushrDriftRL-v0','title':'Continuous MuSHR drifting under random disturbances','steps':250,'hz':50,'family':'drift'},
    'f1tenth-drift': {'id':'Isaac-F1TenthDriftRL-v0','title':'F1Tenth drifting','steps':250,'hz':50,'family':'drift'},
    'elevation': {'id':'Isaac-MushrElevationRL-v0','title':'MuSHR elevation traversal','steps':200,'hz':10,'family':'elevation'},
    'visual': {'id':'Isaac-MushrVisualRL-v0','title':'MuSHR visual navigation','steps':50,'hz':5,'family':'visual'},
}

from third_party.benchmarks.wheeledlab.robotworld.specs import NAMES,scenario
for name in NAMES:
    spec=scenario(name)
    CASES[name]={'id':'RobotWorld-'+name+'-v1','title':spec['title'],'steps':2000,'hz':50,'family':'custom','difficulty':spec['difficulty']}

from third_party.benchmarks.wheeledlab.robotworld.precision_specs import NAMES as PRECISION_NAMES,scenario as precision_scenario
for name in PRECISION_NAMES:
    spec=precision_scenario(name)
    CASES[name]={'id':'RobotWorld-'+name+'-v1','title':spec['title'],'steps':spec['steps'],'hz':50,'family':'custom','precision':True,'difficulty':'hard_target'}

# English model-facing labels; retain original human-facing labels for reports.
MODEL_TITLES = {
    'mushr-drift':'MuSHR continuous drift under random disturbances',
    'f1tenth-drift':'F1Tenth continuous drift under random disturbances',
    'elevation':'MuSHR elevated terrain traversal',
    'visual':'MuSHR visual path following',
    'rw-courtyard':'Courtyard delivery: obstacles, ramp and oriented parking',
    'rw-hairpins':'Elevated narrow bridge and consecutive hairpin turns',
    'rw-gate-dock':'Dynamic gates: yielding, narrow passage and precise docking',
    'rw-drift-switch':'Changing-friction track: sustained drift and disturbance recovery',
    'rw-twin-beam':'Precision twin-beam bridge and oriented parking',
    'rw-reverse-bay':'Narrow reverse bay parking between parked cars',
    'rw-parallel-park':'Tight parallel parking with forward and reverse corrections',
}
