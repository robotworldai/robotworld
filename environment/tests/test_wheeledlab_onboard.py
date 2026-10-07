import json
from types import SimpleNamespace

import pytest

from environment.benchmarks.wheeledlab.custom import CustomSimulator, instructions
from environment.benchmarks.wheeledlab.control import specs
from environment.benchmarks.humanoid_soccer.coding import ControllerProgram
from environment.runtime.events import without_images
from third_party.benchmarks.wheeledlab.robotworld.specs import scenario
from third_party.benchmarks.wheeledlab.robotworld.onboard import telemetry, PROFILE


class Values:
    def __init__(self, values): self.values = values
    def __getitem__(self, _): return self
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.values


def simulator(tmp_path):
    names = ['back_left_wheel_throttle', 'front_left_wheel_steer', 'secret_joint']
    data = SimpleNamespace(joint_pos=Values([0,.1,987654]), joint_vel=Values([10,0,987654]), root_ang_vel_b=Values([.1,.2,.3]))
    robot = SimpleNamespace(joint_names=names, data=data)
    sim = CustomSimulator(SimpleNamespace(step_dt=.02,scene={'robot':robot}), 'rw-gate-dock', tmp_path, scenario('rw-gate-dock'))
    sim.truth = lambda: dict(roll=.02,pitch=.03,position=[987654]*3,yaw=987654,gate={'clear':True})
    sim.terms = {'root_pos_w_term':[987654]*3}
    sim.policy_sensor = SimpleNamespace(image=True,pixels={'type':'image_pixels','width':48,'height':27,'pixels':[128]*3888})
    return sim


def test_custom_public_and_callback_observations_have_no_oracles(tmp_path):
    sim = simulator(tmp_path)
    before = sim.observation()
    assert before['observation_profile'] == PROFILE
    assert set(before) == {'observation_profile','control_step','dt','sensors','last_requested_action','episode_terminated','camera'}
    assert '987654' not in json.dumps(before)
    sim.judge.next = 8; sim.judge.stop_done = True
    assert sim.observation() == before  # Private progress/traffic score cannot leak.
    callback = sim.controller_observation()
    assert set(callback) == set(before) | {'front_camera'}
    assert callback['front_camera']['pixels'] == [128]*3888
    assert '987654' not in json.dumps(callback)


def test_prompt_never_serializes_private_map_or_dynamics(tmp_path):
    sim = simulator(tmp_path)
    for key in ['path','goal','checkpoints','obstacles','friction_bands','perturbations','gate']:
        sim.spec[key] = 'PRIVATE_SENTINEL'
    prompt = instructions(sim)
    assert 'PRIVATE_SENTINEL' not in prompt
    assert 'RED outlined stop box' in prompt
    assert 'front camera' in prompt


def test_telemetry_does_not_substitute_true_ground_speed():
    obs = telemetry(['back_right_wheel_throttle','front_right_wheel_steer'],[0,.2],[14,0],[0,0,.4],.1,.2)
    assert obs['wheel_encoders_rad_s'] == {'back_right_wheel_throttle':14}
    assert obs['steering_encoders_rad'] == {'front_right_wheel_steer':.2}
    assert set(obs['imu']) == {'angular_velocity_body_rad_s','roll_rad','pitch_rad'}
    assert 'velocity' not in obs and 'position' not in obs


def test_controller_can_read_actual_pixels_but_not_global_position(tmp_path):
    obs = simulator(tmp_path).controller_observation()
    program = ControllerProgram("def control(obs,memory):\n return {'action':[obs['front_camera']['pixels'][0]/255,0]}\n")
    try: assert program.request({'obs':obs})['action']['action'][0] == pytest.approx(128/255)
    finally: program.close()
    program = ControllerProgram("def control(obs,memory):\n return {'action':[obs['vehicle']['position'][0],0]}\n")
    try:
        with pytest.raises(ValueError): program.request({'obs':obs})
    finally: program.close()


def test_pixel_payloads_are_removed_only_from_no_images_view():
    source = {'front_camera':{'type':'image_pixels','width':48,'height':27,'pixels':[128]*3888},'action':[.3,.1]}
    result = without_images(source)
    assert isinstance(source['front_camera']['pixels'],list)
    assert 'image pixels omitted' in result['front_camera']['pixels']
    assert result['front_camera']['width'] == 48 and result['action'] == [.3,.1]
    text_result = json.loads(without_images(json.dumps({'final_memory':source})))
    assert 'image pixels omitted' in text_result['final_memory']['front_camera']['pixels']


def test_custom_tool_schemas_describe_sensor_boundary():
    tools = specs(onboard=True)
    assert 'onboard' in tools[0]['description']
    assert 'pixels' in tools[-1]['description']
