"""Isaac 6 conveyor graph compatibility; adapted from local TraceHarness.

Only runtime graph state is changed. Asset files are preserved; when graph
defaults conflict, the explicitly authored physical surface velocity is used.
"""
import math


def scalar_for_surface(direction, surface):
    """Recover node scalar preserving an explicitly authored belt velocity."""
    norm = sum(float(x) ** 2 for x in direction)
    if norm == 0:
        raise ValueError('Conveyor direction is zero')
    scalar = sum(float(a) * float(b) for a, b in zip(direction, surface, strict=True)) / norm
    if not all(math.isclose(float(a) * scalar, float(b), abs_tol=1e-6)
               for a, b in zip(direction, surface, strict=True)):
        raise ValueError('Authored surface velocity is not parallel to node direction')
    return scalar

def initialize_graph_variable(db, name, authored_value):
    if not isinstance(authored_value, (int, float)) or not math.isfinite(authored_value):
        raise ValueError(f'Invalid authored conveyor variable {name}: {authored_value!r}')
    before = db.get_variable(name)
    if before is None:
        raise RuntimeError(f'Conveyor runtime variable missing: {name}')
    db.set_variable(name, authored_value)
    after = db.get_variable(name)
    if not math.isclose(after, authored_value, rel_tol=1e-6, abs_tol=1e-8):
        raise RuntimeError(f'Conveyor variable initialization failed: {name}')
    return before, after


def repair_conveyor_variable_targets(conveyor_asset_path=None):
    """Repair readers and preserve the asset physical surface velocity.

    Conflicting graph defaults are overridden only in runtime, with an audit record.
    """
    import omni.usd
    import omni.graph.core as og
    stage = omni.usd.get_context().get_stage()
    authored_surface = None
    if conveyor_asset_path:
        from pxr import Usd
        source = Usd.Stage.Open(str(conveyor_asset_path))
        candidates = [p.GetAttribute('physxSurfaceVelocity:surfaceVelocity')
                      for p in source.Traverse()
                      if p.GetAttribute('physxSurfaceVelocity:surfaceVelocity')
                      and p.GetAttribute('physxSurfaceVelocity:surfaceVelocity').HasAuthoredValueOpinion()]
        if len(candidates) != 1:
            raise RuntimeError('Expected exactly one authored conveyor surface velocity')
        authored_surface = list(candidates[0].Get())
    repaired = []
    for prim in stage.Traverse():
        if '/dynamic/conveyor/' not in str(prim.GetPath()):
            continue
        if prim.GetAttribute('node:type').Get() != 'omni.graph.core.ReadVariable':
            continue
        node = og.Controller.node(str(prim.GetPath()))
        relation = prim.GetRelationship('inputs:graph')
        paths = relation.GetTargets() if relation else []
        name = prim.GetAttribute('inputs:variableName').Get()
        if len(paths) != 1 or not name:
            continue
        owner = stage.GetPrimAtPath(paths[0])
        variable = owner.GetAttribute('graph:variable:' + name) if owner else None
        if not variable or variable.Get() is None:
            raise RuntimeError(f'Conveyor variable source unavailable for {prim.GetPath()}')
        # Isaac 6 ReadVariable ignores both legacy inputs:graph and targetPath.
        # The live graph variable, not either deprecated input, is its source.
        from types import SimpleNamespace
        graph = node.get_graph()
        context = graph.get_default_graph_context()
        runtime_variable = graph.find_variable(name)
        if not runtime_variable:
            raise RuntimeError(f'Conveyor runtime variable missing: {name}')
        db = SimpleNamespace(
            get_variable=lambda _: runtime_variable.get(context),
            set_variable=lambda _, value: runtime_variable.set(context, value),
        )
        original_authored_value = variable.Get()
        before, after = initialize_graph_variable(db, name, original_authored_value)
        # Recreate the reader through the current API: referenced legacy nodes
        # may retain an unresolved output type and return zero despite a valid variable.
        graph_path = str(owner.GetPath())
        replacement = 'world_read_' + prim.GetName()
        replacement_path = graph_path + '/' + replacement
        consumers = []
        disconnect = []
        surface_override = None
        source = prim.GetPath().AppendProperty('outputs:value')
        for consumer in owner.GetChildren():
            if consumer.GetAttribute('node:type').Get() != 'isaacsim.asset.gen.conveyor.IsaacConveyor':
                continue
            attr = consumer.GetAttribute('inputs:velocity')
            connections = attr.GetConnections()
            if source in connections or (not connections and name == 'Velocity'):
                if authored_surface is not None:
                    direction = consumer.GetAttribute('inputs:direction').Get()
                    desired = scalar_for_surface(direction, authored_surface)
                    if not math.isclose(desired, after, abs_tol=1e-6):
                        surface_override = {'asset': str(conveyor_asset_path),
                                            'authored_surface_velocity': authored_surface,
                                            'node_direction': list(direction),
                                            'graph_scalar_before': after,
                                            'runtime_scalar_after': desired,
                                            'reason': 'Preserve authored physical belt velocity; graph defaults conflict'}
                        db.set_variable(name, desired)
                        # Graph node creation reinitializes variables from the
                        # composed live USD. Author only this stage override;
                        # never save or edit the referenced asset layer.
                        variable.Set(desired)
                        after = db.get_variable(name)
                consumers.append(str(attr.GetPath()))
                if source in connections:
                    disconnect.append((str(source), str(attr.GetPath())))
        if consumers and not stage.GetPrimAtPath(replacement_path):
            keys = og.Controller.Keys
            og.Controller.edit(graph_path, {
                keys.CREATE_NODES: [(replacement, 'omni.graph.core.ReadVariable')],
                keys.SET_VALUES: [(replacement + '.inputs:variableName', name)],
                keys.DISCONNECT: disconnect,
                keys.CONNECT: [(replacement_path + '.outputs:value', dest) for dest in consumers],
            })
        repaired.append({'node': str(prim.GetPath()), 'target': str(paths[0]),
                         'variable': name, 'runtime_before': before,
                         'runtime_after': after, 'authored_value': original_authored_value,
                         'surface_override': surface_override,
                         'replacement_reader': replacement_path, 'consumers': consumers})
    return repaired


def snapshot():
    import omni.usd
    import omni.timeline
    import omni.graph.core as og
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    result = {'playing': timeline.is_playing(), 'timeline_time': timeline.get_current_time(), 'prims': []}
    for prim in stage.Traverse():
        if '/dynamic/conveyor/' not in str(prim.GetPath()):
            continue
        if '/Looks/' in str(prim.GetPath()) or '/Physics_materials/' in str(prim.GetPath()):
            continue
        attrs = {}
        for a in prim.GetAttributes():
            name = a.GetName()
            if name.startswith(('inputs:', 'outputs:', 'graph:variable:', 'physxSurfaceVelocity:', 'physics:')) or name == 'node:type':
                try:
                    value = og.Controller.get(og.Controller.attribute(str(a.GetPath()))) if name.startswith(('inputs:', 'outputs:')) and prim.GetTypeName() == 'OmniGraphNode' else a.Get()
                    attrs[name] = str(value)
                except Exception as exc:
                    attrs[name] = type(exc).__name__
        if attrs:
            result['prims'].append({'path': str(prim.GetPath()), 'attributes': attrs,
                                    'relationships': {r.GetName(): list(map(str, r.GetTargets())) for r in prim.GetRelationships()}})
    return result
