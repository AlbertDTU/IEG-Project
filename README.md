# IEG Project – Integrated Energy Grids

DTU Course 46770 – February 2026

This repository contains coursework for the Integrated Energy Grids project, including assignments 1 & 2. The work applies advanced grid modeling techniques using PyPSA to optimize energy systems at national and regional scales.

## 📋 Project Overview

In this project, you'll develop a comprehensive energy system model for a chosen country or region, optimizing renewable and non-renewable generation capacities while considering interconnections with neighboring systems and energy storage technologies.

---

## 📝 Assignment 1 – System Optimization & Network Analysis

**Deadline:** March 25, 2026, 23:55  
**Format:** Group report (4 students, max 6 pages)  
**Submission:** DTULearn

### Prerequisites
- Review the PyPSA tutorial before starting

### Tasks

#### 1a. Single-Country System Optimization
Choose a country, region, city, or specific energy system and calculate optimal capacities for generators:

- Include both renewable and non-renewable technologies
- Document all cost assumptions and technological parameters with references
- **Analysis required:**
  - Dispatch time series plots for one summer week and one winter week
  - Annual electricity mix breakdown
  - Duration curves or capacity factors showing each technology's contribution

#### 1b. Interannual Weather Sensitivity
Investigate how your results vary year-to-year:

- Run your model using different weather years
- Plot average capacity and variability for all generator types
- Discuss key sensitivities identified

#### 1c. Energy Storage Integration
Add one or more storage technologies and analyze their role:

- Compare system configuration with and without storage
- Evaluate storage behavior and charging/discharging patterns
- Discuss balancing strategies at different timescales:
  - Intraday balancing
  - Seasonal balancing
  - Other relevant timescales

#### 1d. Interconnected Regional Network (DC Approximation)
Expand the system with neighboring countries:

- Connect your country to **at least 3 neighboring countries** via HVAC transmission lines
- Include **at least one closed loop** in the network
- Set transmission capacities based on existing infrastructure data
- Technical assumptions:
  - Voltage level: 400 kV
  - Reactance: x = 0.1
- Optimize the interconnected system using **linearized AC power flow (DC approximation)**
- Discuss results and cross-border power flows

#### 1e. Manual Network Analysis (Pen & Paper)
Replicate simulation results analytically:

- Calculate the **incidence matrix** of your network
- Calculate the **Power Transfer Distribution Factor (PTDF) matrix**
- Extract generation-demand imbalances from the first time step
- Using the PTDF matrix, manually calculate optimal line flows
- Verify results match your PyPSA simulation

---

## 🛠️ Tools & Resources

- **Modeling Framework:** [PyPSA](https://pypsa.io/)
- **Documentation:** Official PyPSA tutorials and documentation
- **Data Sources:** Real grid data for transmission capacities and technology costs

## 📂 Repository Structure

```
IEG Project/
├── README.md                          (this file)
├── Assignment_1/                      (Part 1 materials)
│   ├── models/                        (PyPSA models)
│   ├── analysis/                      (Analysis scripts)
│   └── report/                        (Final report)
└── Assignment_2/                      (Part 2 materials)
```

---

## 📌 Notes

- All cost and technological assumptions must be referenced
- Ensure reproducibility of results
- Include clear visualizations and explanations
- For section 1e, document manual calculations clearly for verification
1