# Replication Code: Angelucci & Prat (2024) – Problem Set 3

Original replication package (Angelucci & Prat, 2024): https://doi.org/10.3886/E195601V1

## Repository Structure
- `run_replication.sh`: Master script to run the replication pipeline.
- `requirements.txt`: Required Python packages and versions.
- `01_prepare.py` – `07_report_tables.py`: Replication, inference, extension, and table/figure scripts.
- `README.md`: Documentation of repository structure, execution order, and data sources.

## Order of Execution & Outputs
1. `01_prepare.py`: Rebuilds the analysis dataset from the raw survey data (sorted file lists, using `bridge_ids3.sav`).
2. `02_estimate_table9.py`: Re-estimates Table 9, column 1 (and checks `bridge_ids1.sav`).
3. `03_table9_inference.py`: Computes corrected confidence intervals and inference checks for Table 9.
4. `04_check_data.py`: Compares the rebuilt dataset against the cleaned data.
5. `05_figure1A.py`: Generates the comparison outputs for Figure 1, panel A.
6. `06_extension_lpm.py`: Estimates the linear probability model extension, fixed effects, OVB checks, and robustness checks.
7. `07_report_tables.py`: Produces the tables/outputs used in the report.

## Notes
- See `requirements.txt` for Python package dependencies.
- Contains our replication code and documentation, not the full original replication package of the authors.
