"""Quick local quality check and daily demand plot."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "raw" / "fingrid_consumption_124.csv"
PLOT = ROOT / "data" / "consumption_daily.png"


def main() -> None:
    if not CSV.exists():
        raise SystemExit("Download data first with: python -m gridwise.ingest")
    df = pd.read_csv(CSV, parse_dates=["start_time_utc", "end_time_utc"])
    if df.empty:
        raise SystemExit("The CSV is empty")
    df = df.sort_values("start_time_utc")
    print(f"Rows: {len(df):,}")
    print(f"First: {df.start_time_utc.iloc[0]}")
    print(f"Last: {df.start_time_utc.iloc[-1]}")
    print(f"Duplicates: {df.start_time_utc.duplicated().sum()}")
    print(f"Missing: {df.isna().sum().to_dict()}")
    print(df.demand_mw.describe().round(1).to_string())
    daily = df.set_index("start_time_utc")["demand_mw"].resample("D").mean()
    ax = daily.plot(figsize=(10, 4), title="Finland electricity demand — daily mean")
    ax.set_ylabel("Average demand (MW)")
    ax.set_xlabel("Date (UTC)")
    plt.tight_layout()
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT, dpi=150)
    plt.close()
    print(f"Saved chart: {PLOT}")


if __name__ == "__main__":
    main()
