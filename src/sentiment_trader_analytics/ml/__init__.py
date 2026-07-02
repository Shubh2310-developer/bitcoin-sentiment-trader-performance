"""Optional machine learning module for the sentiment trader pipeline.

Provides baseline and candidate model pipelines (RandomForest),
chronological train/test splitting, TimeSeriesSplit cross-validation,
evaluation metrics, and feature importance computation.

Only invoked after Phase 06 statistical analysis establishes a
defensible signal worth modelling.
"""
