# Release review — 7 October 2026

This review covers the public source snapshot and its documented entry points. It is not a clean-machine deployment certificate or an all-task simulation validation.

## Documentation

- `README.md` is the English landing page; `README.zh-CN.md` is its Chinese counterpart. Both link to each other.
- Both explain source restoration, assets, Docker, source-built agent runtime, API setup, per-task budgets, code-control settings, repetitions, resume, and output statistics.
- API examples use placeholders and a hidden shell prompt for the key. No user-supplied model credential was used in this review.
- The supported model interface is streaming Responses with image input and function calls. Endpoint reachability is assumed; no deployment-specific routing is required by the examples.

## Source and privacy inspection

Inspected 37,076 tracked files, including bundled upstream code, for token-shaped values, credential assignments, personal deployment paths, private-IP HTTP endpoints, and deployment-specific service identifiers. The available Git history contains one release commit; its files were covered by the tracked-file scan. Remote URL configuration was also checked for embedded credentials.

No project-owned live API credential or personal deployment address was identified. This is a pattern-based inspection with manual classification of matches, not proof that every possible secret format is absent. It does not certify that all deployment metadata has been removed.

Matches retained with their context:

- Upstream Codex tests contain synthetic credentials and private-key fixtures. They are upstream test data; bundled runtime source was not changed.
- Third-party documentation contains upstream author paths, a LAN-address example, and an expired signed media URL from 2024. These are not this deployment's credentials. Third-party attribution, licences, and upstream code were preserved.
- A report-export privacy filter contains a shared-storage path pattern specifically to redact such paths. It is not a configured deployment address.
- Public source, asset and image repository identities remain intentionally visible so users can retrieve dependencies and check provenance.

Generated authentication, caches, outputs, and local build directories are ignored by Git. Review any future generated result bundle separately; source inspection does not sanitise future model traces or credentials supplied later. Do not publish the `.git` directory as an artifact archive.

## Checks performed

| Check | Result |
| --- | --- |
| Existing API configuration/image/tool-round-trip unit tests | 5 passed using a local fake endpoint; not a live provider test. |
| RoboCasa task listing | Passed; 10 task configurations printed. |
| CloseDrawer, one-rollout dry-run | Passed; 450-step task configuration. |
| Full one-rollout campaign dry-run | Passed; 84 launcher commands generated. |
| RoboCasa Docker build-plan dry-run | Passed; actual Docker build not repeated. |
| Full existing Python test suite | 526 passed, 20 failed, 22 skipped using an existing local Python environment. |
| Anonymous pinned HF asset-manifest download | HTTP 401; account access required on the tested path. No full asset download claimed. |
| Live model API and simulator rollout | Pending provider URL, model ID and credential. |

## Open test failures

Do not interpret the full suite as passing. Failures include:

- Assets omitted from source distribution and not yet restored in this directory: RoboCasa base XML, robot USDs, debug-marker resources and legacy runtime files.
- Independent source checkouts not yet restored: revision checks see the outer source-export commit instead of the expected upstream commit.
- Agent Docker preconditions missing in the local test setup.
- Stale catalogue expectations: an AI-CPS test still includes removed task 34; an image-plan test expects 21 tags while the current selected plan has 20.
- Two turn-recovery tests disagree with current content-policy error handling. This requires a separate behaviour decision; it was not silently changed during documentation cleanup.

No success checker, task budget, simulator physics, agent control loop, or upstream source was changed to make these tests pass.

## Next live check

Once a provider is configured, follow the READMEs: configure the provider, run `scripts/check_api.py`, then run one native-budget CloseDrawer rollout. Reuse matching installed images and assets where available. Preserve the run status, events, videos, and model/tool errors; task failure and infrastructure failure must remain distinct. Keep real keys only in private runtime configuration/environment and out of this document.

## Live API follow-up

A user-supplied provider passed the real streaming/image/function-call round-trip check with `gpt-6-astra`. Credentials are stored only in ignored private runtime files, not in source or this report.

The simulator smoke test exposed two deployment defects:

1. The isolated agent did not mount the verified `codex-code-mode-host` helper. Helper mounting is now supported. The pinned helper's V8 archive returned HTTP 404 during this installation attempt; the smoke evaluation instead explicitly uses `WORLD_CODEX_DISABLE_CODE_MODE=1`, including a private direct-tool model catalogue. Codex source and model ID are unchanged.
2. The custom-provider host-isolated path did not invoke request-level image selection. A run reached 63 control steps, then the provider rejected more than 50 accumulated images. The existing trusted-window selector is now attached on this path; traces record actual incoming and outgoing image counts. Original run images are preserved.

After these changes, 48 focused tests passed, including real local HTTP image selection/retry tests, API setup, observation boundaries and direct-tool configuration. The earlier full-suite result is the pre-change baseline; it is not a claim of complete post-change validation. A separate fresh-reset rollout checks the complete simulator path.

### Completed simulator result

The fresh-reset `public-api-smoke-05` rollout finished successfully: RoboCasa `CloseDrawer`, native scoring, seed 7, 407 control steps within the unchanged 450-step limit, environment-side code control off. The runner reported `native_success: true`, `stop_reason: success`, return code 0 and no artifact warnings. This is one successful integration smoke test, not a benchmark-wide performance result.

The request audit contains 22 requests with no recorded violations. Incoming histories reached 303 images; outgoing requests contained at most 15, retaining the trusted observation packet. Full events and video remain in ignored local outputs. The run reused existing pinned source installations, Docker image, assets and a SHA-256-verified app-server; it did not validate a clean-machine rebuild or a complete remote asset download.
