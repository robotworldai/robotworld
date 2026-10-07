"""Visible paint for the sensor-only task; no changed collision or success rule."""
def add_stop_marking(stage, center):
    from pxr import UsdGeom, UsdShade, Gf, Sdf
    material = UsdShade.Material.Define(stage, '/World/RobotWorldStopPaint')
    shader = UsdShade.Shader.Define(stage, '/World/RobotWorldStopPaint/Shader'); shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.85,.08,.04))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    for i, (dx,dy,sx,sy) in enumerate([(-.5,0,.07,1.1),(.5,0,.07,1.1),(0,-.55,1.,.07),(0,.55,1.,.07)]):
        cube = UsdGeom.Cube.Define(stage, f'/World/RobotWorldStopBox/line_{i}')
        cube.CreateSizeAttr(1.); cube.AddTranslateOp().Set(Gf.Vec3d(center[0]+dx,center[1]+dy,center[2]+.012))
        cube.AddScaleOp().Set(Gf.Vec3d(sx,sy,.008)); UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
