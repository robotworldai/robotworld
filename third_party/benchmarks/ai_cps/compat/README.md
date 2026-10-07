# Isaac Sim 6.0.1 External Compatibility Layer

Only API bridge code. `isaac601.py` maps old module names, old torch helper and USD light properties and directs default robots/ground references to original 2022.2.0 downloading assets. Initialization readaptation is also performed in simulator.py class of World. No upstream rewards, terminations, job layouts, collisions USD or joint drive parameters were rewritten. The known collision retreat limit for the experimental operation is found in the upper README and cannot be considered the same as the old physical value.
