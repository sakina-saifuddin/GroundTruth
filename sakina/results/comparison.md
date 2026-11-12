# Experiment comparison

This file is generated from the saved test metrics after each experiment. Missing runs are shown as pending, not estimated.

| Metric | Experiment A (class weights) | Experiment B (oversampling) |
|---|---:|---:|
| Test accuracy | 0.3908 | 0.3952 |
| Test macro-F1 | 0.3051 | 0.3212 |
| Test micro-F1 | 0.3908 | 0.3952 |
| Best val macro-F1 | 0.3464 | 0.3737 |
| Best epoch | 11 | 16 |

## Per-class test F1

| Cause | Experiment A | Experiment B |
|---|---:|---:|
| EQUIPMENT FAILURE | 0.5882 | 0.5822 |
| CORROSION FAILURE | 0.4345 | 0.4972 |
| MATERIAL FAILURE OF PIPE OR WELD | 0.2353 | 0.2832 |
| EXCAVATION DAMAGE | 0.3896 | 0.4000 |
| NATURAL FORCE DAMAGE | 0.2903 | 0.3652 |
| OTHER OUTSIDE FORCE DAMAGE | 0.2740 | 0.2114 |
| OTHER INCIDENT CAUSE | 0.0000 | 0.0000 |
| INCORRECT OPERATION | 0.2286 | 0.2308 |

## Summary

Experiment B (oversampling) has the higher test macro-F1 (0.3212 vs. 0.3051). Compared with class weighting, oversampling had higher test F1 for: NATURAL FORCE DAMAGE (+0.0749), CORROSION FAILURE (+0.0628), MATERIAL FAILURE OF PIPE OR WELD (+0.0479). It had lower test F1 for: OTHER OUTSIDE FORCE DAMAGE (-0.0626), EQUIPMENT FAILURE (-0.0061). These are measured test-set differences, not evidence that oversampling caused the change.
