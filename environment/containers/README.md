# Container configuration

`runtime/` describes the agent-runtime image design. Benchmark directories contain environment recipes, launch templates, and version-compatibility notes. Shared base logic should be reused only where compatibility is established.

Keep adapters separate from upstream source, and mount source, assets, and outputs independently. Record unsupported combinations and approved external compatibility layers explicitly. See the [deployment guide](../../docs/DEPLOYMENT.md) for image prerequisites and build order.
