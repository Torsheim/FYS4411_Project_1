"""Small input/output helpers used by scripts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Any


def write_dicts_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Write a list of dictionaries to CSV."""

    rows = list(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_energy_csv(path: str | Path) -> list[float]:
    """Read an energy CSV created by VMCRunResult.save_energy_csv."""

    values: list[float] = []
    with Path(path).open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            values.append(float(row["energy"]))
    return values
