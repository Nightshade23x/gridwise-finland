# Public demo data bundle

Run `python scripts/export_public_demo.py` from the repository root to generate the
three **real historical CSV snapshots** and `manifest.json` needed for a hosted
read-only demo. The script does not generate synthetic data or include any live
forecast artifacts or Fingrid API credentials.

The CSVs are generated and intentionally not included in this patch. You must
review the generated files before staging them for a public GitHub commit.

Attribution: Fingrid Open Data, consumption dataset 124 (CC BY 4.0).
