# Experiment comparison

This file is generated from the saved test metrics after each experiment. Missing runs are shown as pending, not estimated.

| Metric | Experiment A (class weights) | Experiment B (oversampling) |
|---|---:|---:|
| Test accuracy | 0.5677 | 0.5655 |
| Test macro-F1 | 0.5347 | 0.4681 |
| Test micro-F1 | 0.5677 | 0.5655 |
| Best val macro-F1 | 0.5770 | 0.5604 |
| Best epoch | 22 | 17 |
| Trainable parameters | 903688 | 903688 |

## Per-class test F1

| Cause | Experiment A | Experiment B |
|---|---:|---:|
| EQUIPMENT FAILURE | 0.5854 | 0.6580 |
| CORROSION FAILURE | 0.7293 | 0.6514 |
| MATERIAL FAILURE OF PIPE OR WELD | 0.5536 | 0.5641 |
| EXCAVATION DAMAGE | 0.7890 | 0.7350 |
| NATURAL FORCE DAMAGE | 0.5057 | 0.5405 |
| OTHER OUTSIDE FORCE DAMAGE | 0.5556 | 0.3846 |
| OTHER INCIDENT CAUSE | 0.2449 | 0.0417 |
| INCORRECT OPERATION | 0.3143 | 0.1695 |

## Summary

Experiment A (class weights) has the higher test macro-F1 (0.5347 vs. 0.4681). Compared with class weighting, oversampling had higher test F1 for: EQUIPMENT FAILURE (+0.0726), NATURAL FORCE DAMAGE (+0.0348), MATERIAL FAILURE OF PIPE OR WELD (+0.0105). It had lower test F1 for: OTHER INCIDENT CAUSE (-0.2032), OTHER OUTSIDE FORCE DAMAGE (-0.1709), INCORRECT OPERATION (-0.1448). These are measured test-set differences, not evidence that oversampling caused the change.
