# Environment integration

This package contains RobotWorld's agent and benchmark integrations. Upstream environment implementations live in `third_party/`.

Shared code is organised under `runtime/`, `tools/`, `registry/`, `datasets/`, and `containers/`. Benchmark-specific semantics belong in `benchmarks/<id>/`, robot geometry in `robots/<id>/`, and launch bridges in `integrations/`. Keep benchmark rules outside the agent runtime.

Use the [deployment guide](../docs/DEPLOYMENT.md) and each integration's README and status records to distinguish implemented interfaces from verified simulator runs.

- [Architecture](docs/architecture.md)
- [Contracts](docs/contracts.md)
- [Upstream source policy](docs/upstream-policy.md)
- [Data and containers](docs/data-and-containers.md)
- [Implementation roadmap](docs/roadmap.md)
