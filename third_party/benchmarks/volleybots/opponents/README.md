# Fixed opponent (required; no dummy fallback)

The pinned checkout contains 34 tracked .pt files: 25 br/crossplay 3v3 policies with57-input encoders and9hierarchical skill policies. No compatible complete1v1 opponent has been identified. See checkpoint-audit.json; a37-input Serve skill is not a complete1v1 policy. Before T05 can be scored, place an independently trained/exported policy here and an `opponent.json` with:

```json
{"file":"opponent.pt","sha256":"REAL_SHA256","provenance":"training code/revision/checkpoint/seed/selection procedure","observation_dim":37,"action_dim":4,"action_transform":null}
```

TorchScript interface: tensor[batch,37] -> tensor[batch,4], float32, outputs[-1,1]. It receives exactly player1's native symmetric policy observation, executes at every environment tick, and never receives GPT actions, hidden physics state, reward or task internals. This contract currently supports a feed-forward actor. A recurrent actor needs an explicit state/reset interface extension before use. The weights SHA and provenance are copied to every run.

Do not put a scripted catch controller or a constant-thrust policy here and label it an official benchmark baseline. Choose and disclose the opponent independently of the tested model. Missing or mismatched weights fail before simulator startup.
