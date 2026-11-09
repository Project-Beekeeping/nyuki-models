# Kenyan bee behaviour and harvest-model design

Researched 6 November 2026 for Smart Nyuki. Pilot counties supplied by the owner: Kiambu, Kajiado, Nakuru and Baringo. Hive types: Langstroth and Kenya Top Bar. The owner has no actual telemetry, inspection labels or harvest records yet.

## What the evidence supports

| Observation | Evidence and limits | Design consequence |
| --- | --- | --- |
| Kenyan apiaries span different climates and bee populations | KALRO describes highland, semi-arid, coastal and arid bee populations and regional management differences. County boundaries are too coarse to establish an apiary's environment. | Retain apiary, hive construction, sensor placement and local weather. Do not identify a bee subspecies from county alone. |
| Harvest seasons differ locally | KALRO gives broad dry-spell harvesting guidance and explicitly says timing depends on the region. Local flowering and frame inspection remain necessary. | Calendar is a weak predictor to test, never the readiness label. |
| Even Nakuru has different reported windows | Micheli's 2013 fieldwork records late January–March for Mariashoni; the article cites September–October for Molo from earlier regional literature. These are historical, local observations. | Do not assign one harvest month to all Nakuru hives or claim these are current guarantees. |
| African bees may prioritize growth and respond strongly to disturbance | Kenyan field research describes changing brood/food allocation and absconding after disturbance. A 2025 Kenyan survey includes absconding in colony loss and finds climate-management associations. | Weight changes need colony and intervention context. A sudden loss is an anomaly to investigate, not an automatically identified swarm, theft, absconding event or harvest. |
| Heat, drought and water availability affect colony survival | The 2025 survey covers 589 Kenyan beekeepers; water supplementation is associated with reduced loss in dry/hot conditions. This is observational evidence, not an individual-hive readiness dataset. | Log water availability, shade, feeding and occupancy in field deployment. Simulate dry interruptions and varying shade; do not convert survey associations into hive-level diagnostic thresholds. |
| In-hive humidity is spatial and multifunctional | Research specifically on A. m. scutellata describes constraints and interactions with temperature, gas exchange and water availability. | Use a documented honey-storage sensor position. Air RH or VPD cannot certify honey moisture. |
| Fanning has several functions | Physiological work documents thermal and respiratory functions as well as nectar processing. | Do not label a frequency band as confirmed curing. Add sound only after independent behaviour annotation and an ablation test. |
| Hive design changes what extraction removes | KALRO distinguishes Langstroth supers and returned comb from top-bar comb management. FAO describes movable top-bar comb construction. | Record hive type and actual equipment/comb mass changes. Keep extracted liquid honey yield separate from total mass removed from the hive. |

## Harvest periods found online

These are **context records**, not individual hive events. No artificial exact dates have been invented from month ranges.

| County / locality | Reported timing | Status for Smart Nyuki |
| --- | --- | --- |
| Nakuru, Mariashoni / Mau | Late January–March, recorded in 2013 ethnographic fieldwork | Historical local context; not county-wide, current, or a moisture measurement |
| Nakuru, Molo | September–October, cited in the same article from earlier regional literature | Secondary statement within a primary fieldwork article; weaker evidence |
| Baringo, Pokot study communities | March, August and December reported by respondents in Mwaka's 2014 thesis | Historical local self-reports; hive-level moisture and timestamps are unavailable |
| Kiambu | No sufficiently verified county-specific hive harvest-date series found in this search | Leave unspecified; collect actual apiary records |
| Kajiado | No sufficiently verified county-specific hive harvest-date series found in this search | Leave unspecified; collect actual apiary records |

KALRO's national guidance mentions July–September, November and December–March dry spells while emphasizing regional timing. These ranges overlap and should not be interpreted as a fixed nationwide schedule. In particular, the code does not turn any of these months into `ready=1`.

## The learning problem

The useful question is: **given the telemetry and records available at this moment, how likely is an independently inspected harvest area to meet our stated moisture/capping criteria, and what recoverable yield is supported?**

The implementation separates four quantities:

1. Telemetry validity and recent disturbances.
2. Measured ripeness according to a configurable project policy: moisture at most 18% and capping at least 80%. This is a conservative operational definition, not a universal biological threshold or the general Codex moisture limit.
3. Weighed net extracted yield, observed only after actual extraction. Unharvested inspections are not assigned zero yield.
4. Colony health, reserves and recent feeding, which can block a recommendation independently of ripeness.

XGBoost, CatBoost and LightGBM learn nonlinear interactions in the engineered sensor features. Logistic and prior/median baselines provide useful checks. Probability calibration, abstention and independent held-out tests matter as much as the estimator. The synthetic run selects a candidate; it cannot establish the best field model.

Recent TabPFN research justifies adding a tabular foundation model to a later small-data benchmark. It does not establish that a transformer will learn Kenyan honey ripeness from labels that do not exist. Raw-audio deep learning and sequence models need separately annotated data and tests against the tabular baseline.

## What is implemented, and what still needs observation

Implemented: an hourly synthetic sugar/water mass balance, varying weather/flowering scenarios, dry interruptions, a small local flowering pulse, hive-type features, known intervention corrections, nighttime loss and VPD features, missingness/age checks, probability calibration, chronologically separate training/calibration/selection/test periods, future held-out hive/apiary tests, weighed-yield-only regression, empirical prediction intervals, health/reserve checks, recent-feeding review, and a release check that fails when held-out recommendation precision misses the configured target.

The scheduled decision is at 18:00 Nairobi time, before the simulated inspection disturbance/extraction. This fits KALRO's stated evening harvesting window as an engineering choice, not a requirement that all Kenyan inspections occur at that time. Field deployments with another schedule need a deliberate time contract and recalculated features; records are never silently shifted.

Not established: calibrated Kenyan nectar sugar concentrations, local evaporation coefficients, comb-building costs, a reliable acoustic fanning detector, an absconding diagnosis, a universal colony reserve mass, or county-level model accuracy. The synthetic Langstroth and top-bar scenarios do not reproduce all differences in comb rebuilding or harvesting practices. Actual sensor placement and harvesting procedure must be logged.

Do not open colonies more frequently solely to satisfy the model. Coordinate representative moisture sampling and health/reserve checks with normal beekeeper inspections; use time-stamped external observations where appropriate. The demonstration's weekly inspection cadence is a simulation setting, not a prescription.

## Public datasets considered

| Resource | Potential use | Why it is not Kenyan readiness ground truth |
| --- | --- | --- |
| Senger et al., Germany 2019–2022, 78 colonies | Real weight/temperature/humidity dynamics, interventions, swarm/death events; useful for feature and anomaly work | Different ecology; repository description does not establish paired honey-moisture/capping labels. Publication archive is about 5.6 GB. |
| MSPB, Québec, 53 hives | Multi-sensor features and inspection traits including honey yield; useful for health/yield research | Different climate and bee management; no readiness label can be assumed from yield alone. |
| Healthy Colony Checklist + scale/weather data, North Carolina/Utah | Health gate research with actual inspection protocols | US health data; not Kenyan honey-ripeness validation. |
| Kenya climate/colony-loss survey, 589 beekeepers | Management and climate context | Survey-level losses, not subdaily hive telemetry or moisture measurements. |
| Honey Production Dataset, Mendeley DOI 10.17632/mmcnjdwt3n.2 | Candidate external software benchmark; CC BY 4.0 | Its description gives an `Extract Honey` flag but does not establish how that flag was independently verified against moisture, which hives/locations were observed, or collection provenance. Not accepted as field readiness truth. |

No external dataset has been silently relabeled, mixed into the simulator, or represented as Kenyan observations. Large archives have not been downloaded unnecessarily. Their contracts need inspection before use.

## Sources and applicability

- **KALRO**, Apiculture Farmer Facing TIMPs: hive types, local forage, shade, disturbance, feeding, inspection and harvesting guidance. https://kalrotimps.com/timps/20
- **KALRO**, Apiculture TIMPs: improved top-bar design, separating honey/brood areas and water/forage management. https://www.kalro.org/navcdp/docs/tot_timps/Apiculture-TIMPs-Final-Upload.pdf
- **Micheli (2013)**, *Honey and Beekeeping among the Okiek of Mariashoni, Mau Forest Escarpment, Nakuru District, Kenya*, Ethnorêma 9. Article's printed page 69 discusses the local calendar. https://www.ethnorema.it/wp-content/uploads/N.-9-2013.pdf
- **Mwaka (2014)**, *Bee-keeping and honey production as alternative livelihood strategies among the Pokot of Baringo County, Kenya*. Thesis section 4.2.4.3 contains respondent-reported harvest months. Institutional record: https://kups.ub.uni-koeln.de/8588/ ; author/publication copy: https://www.researchgate.net/publication/299981911
- **University of Nairobi thesis (1992)**, *Resource hoarding and defensive behaviours of African honey bees (Apis mellifera L.) in Kenya*. Historic field evidence; do not extrapolate its disturbance experiments directly to normal inspections. https://erepository.uonbi.ac.ke/items/13478d77-252d-41cd-8d79-82345321dd0f
- **Sibaja Leyton et al. (2025)**, *Climate effects on honey bees can be mitigated by beekeeping management in Kenya*. Journal of Environmental Management 374, 123879. https://doi.org/10.1016/j.jenvman.2024.123879
- **Human et al. (2006)**, *Do honeybees, Apis mellifera scutellata, regulate humidity in their nest?* https://pubmed.ncbi.nlm.nih.gov/16670906/
- **Stabentheiner et al. (2021)**, *Coping with the cold and fighting the heat: thermal homeostasis of a superorganism, the honeybee colony*. https://pmc.ncbi.nlm.nih.gov/articles/PMC8079341/
- **Eyer et al. (2016)**, *A Look into the Cell: Honey Storage in Honey Bees, Apis mellifera*. https://doi.org/10.1371/journal.pone.0161059
- **Meikle et al. (2018)**, *Using within-day hive weight changes to measure environmental effects on honey bee colonies*. https://doi.org/10.1371/journal.pone.0197589
- **FAO**, *Beekeeping in Africa*, top-bar construction and management chapter. https://www.fao.org/4/t0104e/T0104E07.htm
- **Senger et al.**, real sensor/inspection archive. https://zenodo.org/records/10407693
- **MSPB (2024)**, dataset description. https://doi.org/10.1038/s41597-024-03695-1
- **Lower et al. (2025)**, Healthy Colony Checklist dataset. https://data.mendeley.com/datasets/pbbwvjxgbc/1
- **Núñez Velasco (2026)**, Honey Production Dataset, version 2. https://data.mendeley.com/datasets/mmcnjdwt3n/2
- **Hollmann et al. (2025)**, tabular foundation model research. https://doi.org/10.1038/s41586-024-08328-6

These sources inform the design. They do not supply missing Smart Nyuki field measurements or certify the exported synthetic model.
