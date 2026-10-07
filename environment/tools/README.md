# Model tool contracts

The initial motion interface retains `move_eef(targets, note)` and `give_up`. Proposed modules include `move_eef.py`, `give_up.py`, and `schemas/`.

Tools define visible semantics, schemas, and input validation. Robot descriptions and adapters declare supported arms and dimensions. RoboDojo retains its 14 named dimensions and angle conventions. Additional robots must declare compatible semantics; an absolute target must not silently become an increment.

Register new perception, memory, or planning tools with their observation permissions, physics-stepping behaviour, budgets, and serialisation requirements. Scene modification and evaluator ground truth are not default agent capabilities.
