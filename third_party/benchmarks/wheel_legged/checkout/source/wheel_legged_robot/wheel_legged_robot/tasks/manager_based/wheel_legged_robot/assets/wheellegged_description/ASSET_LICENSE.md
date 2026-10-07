# Wheel-legged robot asset attribution and license

## Source

The robot model in this directory is redistributed from:

- Project: [clearlab-sustech/Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym)
- Upstream path: `resources/robots/wl`
- Upstream commit inspected:
  `c354431e5633eb98d9663fa3fbe444201d2de383`

The upstream ROS `package.xml` declares the `wl` description package license as
`BSD`. The upstream repository uses the BSD 3-Clause license. A verbatim copy
of the upstream metadata is included as `package.xml`, and a verbatim copy of
the license is included as `LICENSE-BSD-3-Clause`.

Copyright and attribution remain with the upstream authors and contributors.
Neither their names nor the project name may be used to endorse this derivative
project without prior written permission.

## Included files and local modifications

The following files are byte-for-byte copies of the upstream files:

- `urdf/wl.urdf`
- `meshes/*.STL`

`urdf/wl_dealed.urdf` is a local derivative of the upstream `wl.urdf`. It keeps
the robot geometry, inertial parameters, joint definitions and visual meshes,
but replaces the collision meshes of the four leg links with box collision
primitives for more stable and efficient simulation.

This attribution notice and the accompanying BSD 3-Clause license must be kept
when redistributing these robot assets or their derivatives.
