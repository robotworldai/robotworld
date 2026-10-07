# Rating criteria and scenario design

Browse [Rating Page](../../robotworld-scoring-guide.html). (a) The current list of 84 issues remains in place; AI-CPS34 has been removed from the active question set and batch script; Upstream environmental source and action interfaces remain unchanged; The World external scene layer and running scorer has entered the new determination.

- `task-scoring.json`, `source-extracts.json`: Current task rating, budget and source code evidence.
- `success-proposals.json`: v0.3 has approved the design, 16 scenes, 27 bar status check, each 1 ~ 2 bar.
- `archive/success-proposals-v0.1.json`, `archive/scoring-guide-v0.1.html`: Detailed old version of the proposal, retroactive only.
- [IMPLEMENTATION.md](IMPLEMENTATION.md): Run command, output file, verified range.
- [STATE_CHECK_DESIGN.md](STATE_CHECK_DESIGN.md): Unified status check syntax.

Each issue shows the original target, the new target, the setting of the environment and the simplicity of success. The source of the decision, the two-way map and the English version of prompt are placed in the folding area. Historical prompt and source extracts are identified separately; A new English task attachment is enabled in World v1 running.

The disturbance size/time, site, initial state, command and budget are environmental settings. success only check the end state, end-stage continuous maintenance, or fixed-window state; Processes such as route/jump/bump are created by real trajectory and do not rely on model statements. (a) The original failure and the declared World scenario are kept separately; Configure unexecuted is a running validity issue and does not mix tasks success.

The current status/measure is based on the source code, and there is no explicit World mark for the original hard threshold. The state-of-the-art programme streamlines some of the objectives and converts some of the proportionality requirements to more stringent continuing requirements; Reads through 261 construction status positive and negative and Docker for 16; Complete physical success tracks still need to be validated and not called official equivalents.

Regenerated: `python scripts/build_scoring_guide.py`. page. v0.3 has recorded the design recognition according to the user command; Subsequent revisions save and export JSON independently.

Overall statistics are subject-to-issue rights: SR is calculated first, then 84 is averaged. See [UNIFIED_SCORE.md](../UNIFIED_SCORE.md) for details. The old contact resumes record retention archive, not entering the current question set.
