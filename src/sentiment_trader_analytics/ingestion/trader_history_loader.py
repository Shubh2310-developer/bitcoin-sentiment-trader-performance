"""Loader for the Trader History dataset.

Reads the ``historical_data.csv`` file in chunks (to handle ~45 MB),
enforces dtype coercion (Timestamp → datetime64[UTC], Side/Direction →
categorical, numeric columns → float64, Account → str), attaches
source metadata, and records a SHA-256 checksum for lineage tracking.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_trader_analytics.config import IngestionConfig
from sentiment_trader_analytics.utils.logging_utils import setup_logging

logger = setup_logging(__name__)

REQUIRED_COLUMNS: set[str] = {
    "Account",
    "Timestamp",
    "Side",
    "Direction",
    "Size USD",
    "Execution Price",
    "Closed PnL",
    "Fee",
    "Trade ID",
}


def _compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        file_path: Path to the file to hash.

    Returns:
        The SHA-256 digest as a hexadecimal string.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def _write_lineage(
    output_dir: Path,
    dataset_name: str,
    run_id: str,
    file_path: Path,
    checksum: str,
    row_count: int,
    chunk_count: int,
) -> None:
    """Write a lineage metadata JSON file.

    Args:
        output_dir: Directory to write the lineage file into.
        dataset_name: Dataset name (used in the filename).
        run_id: Unique run identifier.
        file_path: Source file path.
        checksum: SHA-256 digest of the source file.
        row_count: Number of rows loaded.
        chunk_count: Number of chunks read.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lineage = {
        "dataset": dataset_name,
        "run_id": run_id,
        "source_file": str(file_path.resolve()),
        "sha256": checksum,
        "row_count": row_count,
        "chunk_count": chunk_count,
        "ingestion_timestamp_utc": datetime.now(UTC).isoformat(),
        "timestamp": datetime.now(UTC).isoformat(),  # alias for backward-compatible verification
    }
    out_path = output_dir / f"{dataset_name}_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(lineage, f, indent=2)
    logger.info("Lineage written to %s", out_path)


def _estimate_memory(df: pd.DataFrame) -> str:
    """Return a human-readable memory usage estimate.

    Args:
        df: The DataFrame to measure.

    Returns:
        A string like ``"45.2 MB"``.
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    if memory_bytes < 1024**2:
        return f"{memory_bytes / 1024:.1f} KB"
    return f"{memory_bytes / 1024**2:.1f} MB"


def load_trader_history(config: IngestionConfig) -> pd.DataFrame:
    """Load the Trader History dataset from CSV.

    Reads ``config.trader_history_path`` using chunked iteration
    (configurable via ``config.chunk_size``). Enforces dtype coercion:
    ``Timestamp`` (UTC epoch ms) → ``datetime64[UTC]``, ``Side`` /
    ``Direction`` → ``pd.CategoricalDtype``, ``Size USD`` /
    ``Execution Price`` / ``Closed PnL`` / ``Fee`` → ``float64``,
    ``Account`` → ``str``. ``Trade ID`` is preserved as its original
    dtype (string) to avoid precision loss.

    Attaches source metadata to ``df.attrs`` and writes a SHA-256
    lineage file.

    Args:
        config: Ingestion configuration providing the file path,
            chunk size, and lineage output directory.

    Returns:
        A DataFrame with coerced dtypes and source metadata attached.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    file_path = Path(config.trader_history_path)

    if not file_path.exists():
        msg = f"Trader history file not found: {file_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Reading trader history from %s (chunk_size=%d)", file_path, config.chunk_size)

    with open(file_path, "rb") as f:
        header_line = f.readline().decode("utf-8").strip()
    actual_columns = header_line.split(",")
    actual_set: set[str] = set(actual_columns)
    if not REQUIRED_COLUMNS.issubset(actual_set):
        missing = REQUIRED_COLUMNS - actual_set
        msg = f"Trader history CSV missing required columns: {missing}. " f"Found: {actual_columns}"
        logger.error(msg)
        raise ValueError(msg)

    dtype_spec: dict[str, type] = {
        "Account": str,
        "Execution Price": float,
        "Size Tokens": float,
        "Size USD": float,
        "Start Position": float,
        "Closed PnL": float,
        "Fee": float,
        "Trade ID": str,
        "Order ID": str,
    }

    chunks: list[pd.DataFrame] = []
    chunk_count = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=config.chunk_size,
        dtype=dtype_spec,
        keep_default_na=False,
    ):
        chunk_count += 1

        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"].astype(np.float64), unit="ms", utc=True
        )

        chunk["Account"] = chunk["Account"].astype(str)
        chunk["Side"] = chunk["Side"].astype(pd.CategoricalDtype())
        chunk["Direction"] = chunk["Direction"].astype(pd.CategoricalDtype())

        float_cols = ["Size USD", "Execution Price", "Closed PnL", "Fee"]
        for col in float_cols:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype(np.float64)

        if "Leverage" in chunk.columns:
            chunk["Leverage"] = pd.to_numeric(chunk["Leverage"], errors="coerce").astype(np.float64)

        chunks.append(chunk)

    if not chunks:
        msg = f"Trader history file is empty: {file_path}"
        logger.error(msg)
        raise ValueError(msg)

    df = pd.concat(chunks, ignore_index=True)
    total_rows = len(df)
    mem_estimate = _estimate_memory(df)

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    checksum = _compute_sha256(file_path)
    _write_lineage(
        output_dir=Path(config.lineage_output_dir),
        dataset_name="trader_history",
        run_id=run_id,
        file_path=file_path,
        checksum=checksum,
        row_count=total_rows,
        chunk_count=chunk_count,
    )

    df.attrs = {
        "source_file": str(file_path.resolve()),
        "row_count": total_rows,
        "ingestion_timestamp_utc": datetime.now(UTC).isoformat(),
        "chunk_count": chunk_count,
    }

    logger.info(
        "Loaded trader history: path=%s, rows=%d, columns=%s, memory=%s",
        file_path,
        total_rows,
        list(df.columns),
        mem_estimate,
    )

    return df
