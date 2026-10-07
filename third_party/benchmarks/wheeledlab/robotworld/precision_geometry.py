"""Procedural collision meshes for narrow-bridge and parking tests."""
from pathlib import Path


def generate(spec, filename):
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf,Sdf
    filename=Path(filename);filename.parent.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str(filename));UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    root=UsdGeom.Xform.Define(stage,'/Precision');stage.SetDefaultPrim(root.GetPrim())
    friction=UsdShade.Material.Define(stage,'/Precision/Physics')
    mat=UsdPhysics.MaterialAPI.Apply(friction.GetPrim());mat.CreateStaticFrictionAttr(1.2);mat.CreateDynamicFrictionAttr(1.0);mat.CreateRestitutionAttr(0.)
    PhysxSchema.PhysxMaterialAPI.Apply(friction.GetPrim()).CreateFrictionCombineModeAttr('multiply')
    colors={}
    def finish(prim,color,collision):
        color=tuple(color)
        if color not in colors:
            path='/Precision/Looks/m'+str(len(colors));m=UsdShade.Material.Define(stage,path);shader=UsdShade.Shader.Define(stage,path+'/Shader');shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.9)
            m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');colors[color]=m
        binding=UsdShade.MaterialBindingAPI.Apply(prim);binding.Bind(colors[color])
        if collision:
            UsdPhysics.CollisionAPI.Apply(prim);binding.Bind(friction,UsdShade.Tokens.weakerThanDescendants,'physics')
    def cube(name,p,size,color,collision=True):
        obj=UsdGeom.Cube.Define(stage,'/Precision/'+name);obj.CreateSizeAttr(1.);obj.AddTranslateOp().Set(Gf.Vec3d(*p));obj.AddScaleOp().Set(Gf.Vec3d(*size));finish(obj.GetPrim(),color,collision)
    def ramp(name,x0,x1,width,z0,z1):
        top=[(x0,-width/2,z0),(x1,-width/2,z1),(x1,width/2,z1),(x0,width/2,z0)]
        mesh=UsdGeom.Mesh.Define(stage,'/Precision/'+name);mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in top]+[Gf.Vec3f(p[0],p[1],-.7) for p in top])
        mesh.CreateFaceVertexCountsAttr([4]*6);mesh.CreateFaceVertexIndicesAttr([0,1,2,3,7,6,5,4,0,4,5,1,1,5,6,2,2,6,7,3,3,7,4,0]);mesh.CreateSubdivisionSchemeAttr('none')
        finish(mesh.GetPrim(),(.24,.26,.28),True);UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
    cube('Landscape',[0,0,-.85],[18,12,.3],(.24,.3,.22))
    if 'bridge' in spec:
        b=spec['bridge'];z=b['height']
        cube('Approach',[-2,0,-.35],[2,1.8,.7],(.22,.24,.26))
        ramp('Ramp',-1,0,1.8,0,z)
        for i,y in enumerate(b['track_y']):
            cube('Beam_'+str(i),[1.25,y,z-.075],[2.5,b['beam_width'],.15],(.68,.48,.20))
            # Paint only the rail edges, not a supplied centreline oracle.
            for j,side in enumerate([-1,1]):cube(f'Paint/beam_{i}_{j}',[1.25,y+side*(b['beam_width']/2-.006),z+.003],[2.5,.012,.006],(.95,.75,.13),False)
        cube('Landing',[3.6,0,z-.35],[2.2,1.8,.7],(.22,.24,.26))
    else:
        cube('ParkingLot',[0,.15,-.18],[5.6,2.8,.36],(.2,.22,.24))
        xmin,xmax,ymin,ymax=spec['bay']
        for i,(a,b) in enumerate(spec['forbidden_lines']):
            cube('ForbiddenLines/l'+str(i),[(a[0]+b[0])/2,(a[1]+b[1])/2,.009],
                 [max(abs(a[0]-b[0]),spec['line_width']),max(abs(a[1]-b[1]),spec['line_width']),.009],(.95,.85,.12),False)
        for i in range(5):
            cube('Mouth/dash'+str(i),[xmin+(xmax-xmin)*(i+.5)/5,ymax,.009],[(xmax-xmin)/9,.018,.009],(.85,.9,.95),False)
    for item in spec['obstacles']:
        name=item['name'];p=item['center'];size=item['size'];cube('Obstacles/'+name,p,size,item['color'])
        if item.get('car'):
            # Stylized parked car envelope with visible cabin and dark windows;
            # main cuboid is the conservative colliding neighbour envelope.
            cube('Cars/'+name+'/cabin',[p[0],p[1],p[2]+size[2]*.65],[size[0]*.6,size[1]*.6,size[2]*.5],(.10,.16,.20),False)
            along_x=size[0]>=size[1]
            for j,(a,b) in enumerate([(a,b) for a in [-1,1] for b in [-1,1]]):
                wheel=UsdGeom.Cylinder.Define(stage,f'/Precision/Cars/{name}/wheel_{j}')
                wheel.CreateRadiusAttr(.07);wheel.CreateHeightAttr(.045);wheel.CreateAxisAttr('Y' if along_x else 'X')
                dx=a*size[0]*(.33 if along_x else .42);dy=b*size[1]*(.42 if along_x else .33)
                wheel.AddTranslateOp().Set(Gf.Vec3d(p[0]+dx,p[1]+dy,.07));finish(wheel.GetPrim(),(.03,.035,.04),False)
    gx,gy,gz=spec['goal']
    cube('Goal',[gx,gy,gz+.009],[.24,.24,.01],(.10,.65,.20),False)
    for i,(x,y) in enumerate([(-4,-3),(0,4),(4,-3)]):cube('Backdrop/building_'+str(i),[x,y,1],[2,1.4,2.],(.38,.42,.47))
    stage.GetRootLayer().Save();return filename
