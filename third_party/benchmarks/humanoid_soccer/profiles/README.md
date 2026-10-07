# Annotations to fixed assessment configuration

(a) Maintain here the common and publicly available control mission statement of the models; No model evidence, assets or running logs.

`play-soccer-v1.txt` saves as it is the lessons from previous failures of the second round of 1000 measurements, which do not contain action resolution. The same text is obtained when all models run fixed `play-soccer` questions; The result is therefore marked as informed diagnostic, rather than as an unknown case result.

The scenario and running parameters are defined in `World/environment/evaluation/suites.json`: external support for seed=2, geostationary, 1000 control step/20 seconds, training-pitch visual background, direct mode, coding_control + move_joints, ankle-com. The entry point is `scripts/eval/humanoid_soccer.sh`, which is included in the main entrance. Runs to save a copy of this text and the actual complete prompt to the output directory. If this description or fixed protocol is subsequently changed, create and document a new version of profile without mixing different configurations.
