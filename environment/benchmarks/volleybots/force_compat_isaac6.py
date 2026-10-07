"""Preserve the original per-link wrenches with one PhysX articulation submission."""


class NativeWrenchBatch:
    def __init__(self, env):
        import torch
        self.torch = torch
        self.env = env
        self.articulation = env.drone._view._physics_view
        self.names = list(env.drone._view._body_names)
        if env.num_envs != 1 or self.articulation.count not in (1,2) or len(self.names) != self.articulation.max_links or len(set(self.names)) != len(self.names):
            raise RuntimeError('Wrench compatibility is audited only for one or two articulations with unique link names')
        self.force = torch.zeros((self.articulation.count,self.articulation.max_links,3),device=env.device)
        self.torque = torch.zeros_like(self.force)
        self.indices = torch.arange(self.articulation.count,device=env.device)
        self.seen = set()
        self.views = {'rotors':env.drone.rotors_view,'base':env.drone.base_link}
        self.link_paths = [p for paths in self.articulation.link_paths for p in paths]
        self.locations = {p:(i,j) for i,paths in enumerate(self.articulation.link_paths) for j,p in enumerate(paths)}
        for view in self.views.values():
            for path in view._physics_view.prim_paths:
                if path not in self.link_paths or path.rsplit('/',1)[-1] not in self.names:
                    raise RuntimeError(f'Wrench body does not belong to the audited articulation: {path}')
        for key, view in self.views.items():
            view._physics_view.apply_forces_and_torques_at_position = (
                lambda force_data, torque_data, position_data, indices, is_global, _key=key:
                    self.queue(_key,force_data,torque_data,position_data,indices,is_global))

    def queue(self, key, force_data, torque_data, position_data, indices, is_global):
        from volley_bots.utils.torch import quat_rotate
        # These two fixed upstream tasks apply at each link origin. Refuse an
        # unknown offset rather than silently changing the resulting moment.
        if position_data is not None:
            raise RuntimeError('Unaudited explicit wrench application position')
        view = self.views[key]._physics_view
        selected = indices.detach().cpu().tolist()
        body_indices = [self.locations[view.prim_paths[i]] for i in selected]
        if self.seen.intersection(body_indices):
            raise RuntimeError('Unexpected repeated per-link wrench in one native physics tick')
        self.seen.update(body_indices)
        q = view.get_transforms()[:,[6,3,4,5]]
        for values, target in ((force_data,self.force),(torque_data,self.torque)):
            if values is None:
                continue
            world = values if is_global else quat_rotate(q,values)
            for source_index, body_index in zip(selected,body_indices):
                target[body_index] = world[source_index]
        return True

    def submit(self):
        # Submit the complete array, including zero entries, exactly once.
        # This prevents a later RigidBodyView packet from clearing earlier
        # forces on other links of this same articulation in the new backend.
        # Keep the existing single-aircraft diagnostic schema unchanged.
        single = self.articulation.count == 1
        self.last_submission = {'body_names':self.names,'link_paths':self.link_paths,
            'world_force':(self.force[0] if single else self.force).detach().cpu().tolist(),
            'world_torque':(self.torque[0] if single else self.torque).detach().cpu().tolist(),
            'application_point':'each original link transform origin',
            'native_body_indices_written':sorted(j for _,j in self.seen) if single else sorted(self.seen)}
        self.articulation.apply_forces_and_torques_at_position(
            self.force,self.torque,None,self.indices,True)

    def clear(self):
        self.force.zero_()
        self.torque.zero_()
        self.seen.clear()
