# IEG Project – Integrated Energy Grids in the Nordics

DTU Course 46770 — Group 5 — May 2026

This repository contains the full course project for 46770 Integrated Energy Grids. The project builds a comprehensive energy system model for the Nordic region (Denmark, Sweden, Norway, Germany) using PyPSA, covering optimal capacity planning, storage, interconnection, sector coupling, and gas networks.

---

## Setup

All scripts run in the `IEG_env` conda environment (Python 3.14.3). A licensed Gurobi solver is required.

```bash
conda activate IEG_env
python part_a/part_a.py
```

For Jupyter notebooks, select the **Python (IEG_env)** kernel in VSCode.

---

## Repository Structure

```
IEG Project/
├── data/                  Input data (CSV files — see Data section below)
├── part_a/                Task a: single-country optimisation
├── part_b/                Task b: interannual variability
├── part_c/                Task c: battery storage
├── part_d/                Task d: multi-country DC network
├── part_e/                Task e: incidence + PTDF matrix (LaTeX)
├── part_f/                Task f: CO2 sensitivity sweep
├── part_g/                Task g: gas pipeline network
├── part_h/                Task h: CO2 shadow price
├── part_i/                Task i: sector coupling (electricity + heat)
├── part_j/                Task j: research question sensitivity analysis
├── plots_part_f/          Output figures and CSVs for task f
└── plots_part_i/          Output figures and CSVs for task i
```

---

## Tasks

| File | Task | Description |
|------|------|-------------|
| `part_a/part_a.py` | a | Single-country (DNK) optimal capacity mix for renewable and non-renewable generators; dispatch time series, annual electricity mix, and duration curves |
| `part_b/part_b.py` | b | Interannual weather sensitivity across 9 weather years (1979–2015); average capacities and variability |
| `part_c/part_c.py` | c | Battery storage impact; compares optimal system with and without storage |
| `part_d/part_d.py` | d | Multi-country DC power flow (DNK, SWE, NOR, DEU) with pumped hydro and fixed HVDC transmission capacities |
| `part_e/part_e.tex` | e | Manual derivation of incidence matrix and PTDF matrix; verification of power flows against PyPSA results |
| `part_f/part_f.py` + `part_f.ipynb` | f | CO2 constraint sensitivity sweep (15 scenarios, 9 Mt to 0); two-section notebook architecture separates optimisation from plotting |
| `part_g/Part_g.py` | g | Multi-country gas pipeline network (CH4); compares energy transported by electricity vs. gas networks |
| `part_h/part_h.py` | h | CO2 shadow price at 50% decarbonisation target; comparison against current EU ETS carbon price |
| `part_i/part_i.py` | i | Sector coupling: co-optimisation of electricity and heating sectors (heat pumps + gas boilers) across all four countries |
| `part_j/part_j.py` | j | Research question: sensitivity analysis across 12 scenarios combining gas price (30–120 EUR/MWh) and offshore wind capital cost reductions (-20% to -60%) |

---

## Data

All input data is in the `data/` folder.

| File | Description |
|------|-------------|
| `electricity_demand.csv` | Hourly MWh electricity demand per country (1979–2017) |
| `onshore_wind_1979-2017.csv` | Hourly onshore wind capacity factors |
| `offshore_wind_1979-2017.csv` | Hourly offshore wind capacity factors |
| `pv_optimal.csv` | Hourly solar PV capacity factors |
| `heat_demand.csv` | Hourly heating demand per country (sector coupling) |
| `temperature.csv` | Hourly temperature data (for heat pump COP calculation) |
| `nodes.csv`, `edges.csv` | Network topology for multi-country tasks |

---

## Tools

- **Modelling framework:** [PyPSA](https://pypsa.io/)
- **Solver:** Gurobi
- **Language:** Python 3.14.3 (`IEG_env` conda environment)
