"""Spectator-only room backdrop: hidden outside review rendering, no physics APIs."""
from contextlib import contextmanager
import json
from pathlib import Path


class ReviewBackdrop:
    def __init__(self, output, center, wall_height=5.):
        import omni.usd
        from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema
        self.stage = omni.usd.get_context().get_stage()
        self.root = UsdGeom.Xform.Define(self.stage, '/World/WorldReviewBackdrop')
        self.visibility = UsdGeom.Imageable(self.root).CreateVisibilityAttr('invisible')
        x, y = float(center[0]), float(center[1])
        def box(name, pos, size, color):
            prim = UsdGeom.Cube.Define(self.stage, '/World/WorldReviewBackdrop/'+name)
            prim.CreateSizeAttr(1.)
            prim.AddTranslateOp().Set(Gf.Vec3d(*pos))
            prim.AddScaleOp().Set(Gf.Vec3f(*size))
            prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        # A distant lab/sports-hall wall supplies scale and depth without covering
        # the original ground, collision surfaces, task objects or terrain.
        box('rear_wall', (x,y+8,wall_height/2), (24,.08,wall_height), (.55,.61,.66))
        box('left_wall', (x-8,y,wall_height/2), (.08,16,wall_height), (.46,.53,.59))
        box('rear_lower_band', (x,y+7.94,.6), (24,.025,1.2), (.18,.27,.32))
        for i in range(-3,4):
            box(f'window_{i+3}', (x+3*i,y+7.93,3.1), (1.9,.025,1.5), (.26,.39,.47))
            box(f'column_{i+3}', (x+3*i+1.4,y+7.90,wall_height/2), (.12,.12,wall_height), (.68,.70,.70))
        for prim in self.stage.Traverse():
            if str(prim.GetPath()).startswith('/World/WorldReviewBackdrop'):
                if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise RuntimeError('Review decoration must not have physics APIs')
        self.output = Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        (self.output/'review-scenery.json').write_text(json.dumps({
            'style':'distant lab/sports hall', 'spectator_only':True,
            'physics_apis':False, 'hidden_outside_review_render':True,
            'original_ground_and_task_geometry_unchanged':True,
            'anchor_xy':[x,y], 'wall_height':wall_height,'decorative_not_task_evidence':True},indent=2))

    @contextmanager
    def visible(self):
        self.visibility.Set('inherited')
        try:
            yield
        finally:
            self.visibility.Set('invisible')


def camera_offsets(task):
    # Fit the whole embodiment, not just the torso or drone body.
    if task in ('T07','T08'):
        return (1.15,-1.45,.85), (0,0,-.03)
    if task in ('T09','T10','T11'):
        return (1.55,-1.8,1.1), (0,0,-.08)
    if task in ('T01','T13','T15'):
        return (2.,-2.4,1.35), (0,0,-.25)
    return (2.5,-2.5,2.), (0,0,0)  # Cup/ball task keeps interception workspace.


@contextmanager
def aerial_review(sim, other='payload'):
    """Frame aircraft plus payload/ball together; extra state is spectator-only."""
    from isaacsim.core.utils.viewports import set_camera_view
    drone = sim.env.drone.get_world_poses()[0].reshape(-1,3)[0].detach().cpu().tolist()
    obj = getattr(sim.env,other).get_world_poses()[0].reshape(-1,3)[0].detach().cpu().tolist()
    center = [(a+b)/2 for a,b in zip(drone,obj)]
    span = sum((a-b)**2 for a,b in zip(drone,obj))**.5
    # FOV fits the entire 1m rod; widen if the volleyball separates further.
    distance = max(2.6, span*1.5)
    if not hasattr(sim,'review_backdrop'):
        sim.review_backdrop = ReviewBackdrop(sim.output,center,getattr(sim,'review_wall_height',5.))
    set_camera_view(eye=[center[0]+distance*.65,center[1]-distance,center[2]+distance*.35],target=center)
    with sim.review_backdrop.visible():
        yield
