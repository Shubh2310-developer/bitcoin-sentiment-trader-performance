"""Data ingestion layer for the sentiment trader analytics pipeline.

Provides loaders for the Fear & Greed Index and Trader History datasets.
Each loader reads raw CSV files, enforces dtype coercion, attaches source
metadata, and records SHA-256 lineage checksums.
"""
