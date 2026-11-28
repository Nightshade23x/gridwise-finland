"""Validate saved Phase 22 reports without importing forecasting or ML dependencies.

The FastAPI image intentionally installs only ``.[app]``; keep this file
standard-library-only. Issuance and verification still run in the ML environment.
"""
from __future__ import annotations

import math

HORIZON_STEPS = 96
STATUSES = frozenset({"pending", "partial", "complete", "incomplete_source_data"})


def validate_report(report: dict) -> None:
    """Reject malformed verification reports before exposing them through the API."""
    if (not isinstance(report, dict) or report.get("schema_version") != 1 or
            report.get("status") not in {"ok", "no_issuances"} or
            not isinstance(report.get("issuances"), list) or
            report.get("issuance_count") != len(report["issuances"])):
        raise ValueError("Invalid shadow verification report metadata")
    for record in report["issuances"]:
        if record.get("status") not in STATUSES:
            raise ValueError("Invalid verification status")
        if not (0 <= record["matched_targets"] <= record["eligible_targets"] <= HORIZON_STEPS and
                record["corrected"]["paired_rows"] == record["uncorrected"]["paired_rows"] == record["matched_targets"]):
            raise ValueError("Invalid verification counts")
        for name in ("uncorrected", "corrected"):
            for metric in ("mae_mw", "rmse_mw", "bias_mw"):
                val = record[name][metric]
                if val is not None and not math.isfinite(float(val)):
                    raise ValueError("Invalid verification score")
