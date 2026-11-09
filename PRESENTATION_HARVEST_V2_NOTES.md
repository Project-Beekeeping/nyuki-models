# Harvest v2 presentation notes

Use the executed v2 notebook and outputs/harvest_v2/RESULTS_SUMMARY.md for the updated harvest demonstration. The existing PowerPoint files have not been edited by this implementation.

## Replace the old harvest claims

- Remove the old 1.7 kg / 8 days / 79% figures unless independently reproduced and identified as a separate experiment.
- Describe this task as present, inspection-defined readiness plus recoverable event yield. Precise days-to-harvest and a trained 14-day dearth classifier are not outputs of v2.
- State that readiness labels require representative moisture/capping observations, not harvest months copied from the internet.
- Show the full test table with sample counts and recall. Pooled precision hides weaker and sparse hive-type results.
- Show the blocked release check and inspect_hive output. Do not describe this as a deployed or field-validated harvesting system.

## Suggested spoken explanation

“We built a reproducible pipeline for Kenyan Langstroth and top-bar scenarios. It compares calibrated readiness models and yield models using separate chronological development and held-out hive/apiary groups. The simpler readiness baseline won this experiment, while XGBoost won yield selection. The release check still blocks advice because pooled scores hide weaknesses by hive type. All current results are synthetic; actual moisture, inspection and weighed-harvest records will establish field performance.”

## Local context

Pilot counties: Kiambu, Kajiado, Nakuru and Baringo. Historical sources provide local seasonal windows in parts of Baringo and Nakuru; the search did not establish a hive-level harvest-date series for all four counties. Treat those records as context. See KENYA_BEE_BEHAVIOUR_AND_MODEL_DESIGN.md and data_templates/county_harvest_context.csv for sources and evidence limits.
