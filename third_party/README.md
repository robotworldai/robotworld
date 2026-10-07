# Third Party Source

`benchmarks/` with original benchmark source code, `dependencies/` with IsaacLab, CuRobo, etc. Codex uses `World/codex/`. adapter, compatibility layer, tool description with `environment/`, without modifications upstream.

(a) A warehouse has been set up with `git clone --local` to create clean work tree without downloading the network and not copying untraceed assets/cache/output; Git non-variable objects can be shared with hard links, the work tree file is independent and does not use alternates which relies on the old path. Version and original source see `sources.local.json`. The user repository does not move or reset. Failure to submit changes will not be disguised as an official source copy.

Source code checkout does not default to the parent repository; Release metadata retains the URL, commit, licence, and retrieval instructions; submodules can be registered explicitly later. Do not put assets, keys, Docker tar here.
