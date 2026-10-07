"""Diagnostic A/B experiments only; never a benchmark/model evaluation."""
import json
from pathlib import Path
import sys

from .project_isaac6 import create_sim


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    sim = create_sim('T16', 7, output)
    torch = sim.torch
    drone = sim.env.drone
    articulation = drone._view._physics_view
    names = list(drone._view._body_names)
    report = {'diagnostic_only': True, 'body_names': names,
              'masses': articulation.get_masses().cpu().tolist(),
              'inertias': articulation.get_inertias().cpu().tolist(), 'cases': {}}
    views = {'rotors':drone.rotors_view, 'base':drone.base_link, 'payload':sim.env.payload}
    originals = {key:view._physics_view.apply_forces_and_torques_at_position for key,view in views.items()}
    active = {'mode':'normal'}
    force = torch.zeros((1, articulation.max_links, 3), device=sim.env.device)
    torque = torch.zeros_like(force)
    from omni_drones.utils.torch import quat_rotate

    def dispatch(key, force_data, torque_data, position_data, indices, is_global):
        mode = active['mode']
        if mode == 'ballistic' or (mode == 'rotors_only' and key != 'rotors'):
            return True
        if mode != 'batched_articulation':
            return originals[key](force_data, torque_data, position_data, indices, is_global)
        if position_data is not None:
            raise RuntimeError('Unexpected non-origin force in this diagnostic')
        view = views[key]._physics_view
        q = view.get_transforms()[:, [6, 3, 4, 5]]
        for values, dest in ((force_data, force), (torque_data, torque)):
            if values is None:
                continue
            converted = values if is_global else quat_rotate(q, values)
            for index in indices.cpu().tolist():
                name = view.prim_paths[index].rsplit('/', 1)[-1]
                dest[0, names.index(name)] += converted[index]
        return True

    for key, view in views.items():
        view._physics_view.apply_forces_and_torques_at_position = (
            lambda force_data, torque_data, position_data, indices, is_global, _key=key:
                dispatch(_key,force_data,torque_data,position_data,indices,is_global))
    original_step = sim.env.sim.step
    def step(render=True):
        if active['mode'] == 'batched_articulation':
            articulation.apply_forces_and_torques_at_position(
                force, torque, None, torch.tensor([0],device=sim.env.device), True)
        original_step(render)
        force.zero_(); torque.zero_()
    sim.env.sim.step = step
    try:
        for mode in ('normal', 'ballistic', 'rotors_only', 'batched_articulation'):
            active['mode'] = mode
            sim.reset(7)
            rows = []
            for _ in range(10):
                if sim.done:
                    break
                before = sim.observation()
                sim.step([.8,.8,.8,.8])
                rows.append({'before':before,'after':sim.observation(), 'evaluation':sim.last_evaluation})
            report['cases'][mode] = rows
            (output/'force-diagnostic.json').write_text(json.dumps(report,indent=2))
            print('[force diagnostic]',mode,'steps',len(rows),'last velocity',
                  rows[-1]['after']['native_policy_observation'][10:16],flush=True)
    finally:
        sim.close()


if __name__ == '__main__':
    main()
