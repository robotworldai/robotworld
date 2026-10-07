"""Procedural USD road, terrain and industrial scenery; no downloaded assets."""
import math
from pathlib import Path

def generate(spec,filename):
    from pxr import Usd,UsdGeom,UsdPhysics,Gf,PhysxSchema,UsdShade
    stage=Usd.Stage.CreateNew(str(filename));UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
    root=UsdGeom.Xform.Define(stage,'/Course');stage.SetDefaultPrim(root.GetPrim())
    def material(name,friction):
        m=UsdShade.Material.Define(stage,'/Course/Materials/'+name);p=UsdPhysics.MaterialAPI.Apply(m.GetPrim());p.CreateStaticFrictionAttr(friction);p.CreateDynamicFrictionAttr(friction);p.CreateRestitutionAttr(0.)
        PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim()).CreateFrictionCombineModeAttr('multiply');return m
    mats=[material('road_'+str(i),f) for i,f in enumerate(spec.get('friction_bands',[1.0]))]
    def bind(prim,mat):UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics')
    looks={}
    def appearance(prim,color):
        key=tuple(color)
        if key not in looks:
            name='/Course/Looks/material_'+str(len(looks));m=UsdShade.Material.Define(stage,name);shader=UsdShade.Shader.Define(stage,name+'/Shader');shader.CreateIdAttr('UsdPreviewSurface')
            from pxr import Sdf
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.85);m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');looks[key]=m
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(looks[key])
    def cube(path,center,size,color,collision=True):
        obj=UsdGeom.Cube.Define(stage,'/Course/'+path);obj.CreateSizeAttr(1.);obj.AddTranslateOp().Set(Gf.Vec3d(*center));obj.AddScaleOp().Set(Gf.Vec3d(*size));obj.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        appearance(obj.GetPrim(),color)
        if collision:UsdPhysics.CollisionAPI.Apply(obj.GetPrim());bind(obj.GetPrim(),mats[0])
        return obj
    cube('Landscape',[0,1,-.95],[34,27,.5],[.30,.34,.27])
    path=[list(p) for p in spec['path']];closed=math.dist(path[0],path[-1])<1e-6
    if not closed:
        a,b=path[:2];dx,dy=b[0]-a[0],b[1]-a[1];n=math.hypot(dx,dy);path.insert(0,[a[0]-1.2*dx/n,a[1]-1.2*dy/n,a[2]])
        a,b=path[-2:];dx,dy=b[0]-a[0],b[1]-a[1];n=math.hypot(dx,dy);path.append([b[0]+1.2*dx/n,b[1]+1.2*dy/n,b[2]])
    edge=[];width=spec['width']/2
    for i,p in enumerate(path):
        prev=path[i-1] if i else path[-2] if closed else path[0]
        nxt=path[i+1] if i+1<len(path) else path[1] if closed else path[-1]
        def normal(a,b):
            dx,dy=b[0]-a[0],b[1]-a[1];n=math.hypot(dx,dy);return (-dy/n,dx/n) if n else None
        n1=normal(prev,p);n2=normal(p,nxt);n1=n1 or n2;n2=n2 or n1
        m=[n1[0]+n2[0],n1[1]+n2[1]];mag=math.hypot(*m);m=[v/mag for v in m];scale=width/max(.6,m[0]*n2[0]+m[1]*n2[1])
        edge.append(([p[0]+m[0]*scale,p[1]+m[1]*scale,p[2]],[p[0]-m[0]*scale,p[1]-m[1]*scale,p[2]]))
    for i in range(len(edge)-1):
        a,b=edge[i];c,d=edge[i+1];top=[a,b,d,c];bottom=[[p[0],p[1],-.7] for p in top]
        mesh=UsdGeom.Mesh.Define(stage,f'/Course/Road/segment_{i:03d}');mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in top+bottom]);mesh.CreateFaceVertexCountsAttr([4]*6)
        mesh.CreateFaceVertexIndicesAttr([0,1,2,3,7,6,5,4,0,4,5,1,1,5,6,2,2,6,7,3,3,7,4,0]);mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDisplayColorAttr([Gf.Vec3f(.17+.025*(i%2),.19,.21)])
        appearance(mesh.GetPrim(),(.17+.025*(i%2),.19,.21))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none');bind(mesh.GetPrim(),mats[min(len(mats)-1,i*len(mats)//(len(edge)-1))])
        # Visual edge reflectors, outside the allowed road footprint, no extra collisions.
        if i%2==0:
            for j,p in enumerate([a,b]):cube(f'RoadMarkers/m_{i}_{j}',[p[0],p[1],p[2]+.035],[.12,.12,.07],[.95,.75,.13],False)
    for o in spec['obstacles']:cube('Obstacles/'+o['name'],o['center'],o['size'],o['color'])
    # Backdrop geometry stays outside all routes. Buildings have real colliders.
    for i,(x,y,sx,sy,h) in enumerate([(-10,2,2,6,2.2),(0,-8,6,2,2.6),(10,1,2,5,2.8),(-1,10,8,2,2.4)]):
        cube(f'Scenery/building_{i}',[x,y,-.7+h/2],[sx,sy,h],[.42+.05*i,.45+.03*i,.48+.02*i])
        cube(f'Scenery/roof_{i}',[x,y,-.7+h+.06],[sx+.15,sy+.15,.12],[.18,.22,.26],False)
        for j in range(4):cube(f'Scenery/window_{i}_{j}',[x+(j-1.5)*sx/4,y-sy/2-.012,-.7+h*.65],[sx/6,.025,.48],[.12,.24,.32],False)
    for k,checkpoint in enumerate(spec['checkpoints']):
        p=checkpoint['position'];cube(f'Checkpoints/stripe_{k}',[p[0],p[1],p[2]+.006],[.14,.14,.009],[.1,.65,.86],False)
    goal=spec['goal'];cube('Goal/marker',[goal[0],goal[1],goal[2]+.009],[.7,.7,.012],[.15,.65,.27],False)
    stage.GetRootLayer().Save()
    return Path(filename)
