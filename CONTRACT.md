# PathSnap Data Contract

## Dataset partitions
- train: 80 trips
- student_test: 10 trips
- Evaluator data is excluded from development.

## Model-ready files
- data/features_train.parquet
- data/features_student_test.parquet

Each row represents one GPS observation–candidate road arc pair.

## Identifiers and target
- trip_id: trip identifier; use for group-aware splitting.
- obs_idx: GPS observation index.
- arc_id: candidate road arc ID.
- is_ref: binary reference label; training target only.

## Model input features
1. perp_distance_m
2. projection_fraction
3. arc_length_m
4. heading_difference_deg
5. candidate_rank
6. candidate_count
7. distance_gap_m

Features are standardized using a StandardScaler fitted on training rows only.

## Leakage restrictions
Never use ref_arc, is_ref, route IDs, or other ground-truth information as model inputs.
Keep trip_id for grouping, not as a predictive feature.
Do not use student_test labels for fitting or tuning.

## Candidate generation
- Densified road-segment spatial index.
- Sample spacing: 10 metres.
- Selected candidate radius: 50 metres.
- The 2% route-set miss-rate target was not achieved at any tested radius.
- Reference labels are assigned to the nearest generated candidate belonging to the trip's route arc set.
- This is a spatial route-set proxy, not a time-aligned ground-truth match.

## Reproducibility
- Random seed: 42
- Projection: per-trip UTM CRS.
- Scaler: models/artifacts/scaler.joblib
- Feature list: models/artifacts/feature_list.json
- Configuration: models/artifacts/config.json
