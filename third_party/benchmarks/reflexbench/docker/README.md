# Docker

`Dockerfile` reuse `world/wheeledlab:isaac6.0.1-experimental`; Add mirror name
`world/reflexbench:isaac6.0.1-experimental`。 Do not copy asset or model login information into the mirror.

```bash
docker build -t world/reflexbench:isaac6.0.1-experimental third_party/benchmarks/reflexbench/docker
```

Version boundary: Isaac Sim 6.0.1/ IsaacLab 2.2; Newer upstream catch mission `sim.utils.prims.clone`
Compatible by World external module. Environment-related realization is still imported from `checkout/`; The code tool does not have access to this path.
