# Agent-runtime image design

A runtime image should build from the pinned `codex/` source or include a locally built binary whose provenance has been checked. Record the dependency lock, binary hash, and source commit. Installing a different CLI through npm or pip is not an equivalent evaluation runtime.

Inject configuration and credentials at runtime. Images must not include personal authentication or global installation files. The benchmark coordinator communicates through the standard app-server protocol.
