import os
import pandas as pd
import pypsa

# =============================================================================
# 1. DATA LOADING
# =============================================================================
df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)
df_elec.index = pd.to_datetime(df_elec.index)

df_onshorewind = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0)
df_onshorewind.index = pd.to_datetime(df_onshorewind.index)

df_offshorewind = pd.read_csv('data/offshore_wind_1979-2017.csv', sep=';', index_col=0)
df_offshorewind.index = pd.to_datetime(df_offshorewind.index)

df_solar = pd.read_csv('data/pv_optimal.csv', sep=';', index_col=0)
df_solar.index = pd.to_datetime(df_solar.index)

df_heat = pd.read_csv('data/heat_demand.csv', sep=';', index_col=0)
df_heat.index = pd.to_datetime(df_heat.index)

# Temperature data: mixed formats (hourly ISO8601 + some daily rows), needs cleaning
df_temp = pd.read_csv('data/temperature_20260429.csv', sep=';', index_col=0)
if 'time' in df_temp.index:
    df_temp = df_temp.drop('time')
# Parse with utc=True so all timestamps become UTC-aware, then strip to naive
df_temp.index = pd.to_datetime(df_temp.index, format='mixed', utc=True)
df_temp.index = df_temp.index.tz_localize(None)  # make naive to align with snapshots
df_temp = df_temp.loc[~df_temp.index.duplicated(keep='first')].sort_index()
df_temp = df_temp[~df_temp.index.isna()]  # drop NaT rows that break monotonic check
for col in df_temp.columns:
    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
df_temp = df_temp.ffill().bfill()

# =============================================================================
# 2. NETWORK INITIALISATION
# =============================================================================
network = pypsa.Network()
hours_in_2015 = pd.date_range('2015-01-01 00:00Z', '2015-12-31 23:00Z', freq='h')
network.set_snapshots(hours_in_2015.values)

# Naive hourly snapshots for temperature and hydro alignment (snapshots.values strips UTC)
naive_snapshots = pd.date_range('2015-01-01', periods=8760, freq='h')

countries = ["DNK", "SWE", "NOR", "DEU"]
gas_price  = 30.0  # €/MWh_th

def annuity(n, r):
    return r / (1. - 1. / (1. + r) ** n) if r > 0 else 1 / n

# =============================================================================
# 3. BUSES, ELECTRICITY LOADS, HEAT BUSES, HEAT LOADS
# Note: use .values directly — df_elec and df_heat cover exactly 2015 in order.
#       .reindex(network.snapshots) fails because .values strips UTC from snapshots.
# =============================================================================
for c in countries:
    network.add("Bus", c, v_nom=400)
    network.add("Bus", f"{c} heat", carrier="heat")
    network.add("Load", f"electricity_demand_{c}", bus=c,
                p_set=df_elec[c].values)
    network.add("Load", f"heat_demand_{c}", bus=f"{c} heat",
                p_set=df_heat[c].values)

# =============================================================================
# 4. CARRIERS
# =============================================================================
for carrier, co2 in [
    ("gas", 0.19), ("onshorewind", 0), ("offshorewind", 0),
    ("solar", 0), ("battery storage", 0), ("pumped hydro", 0),
    ("heat", 0), ("heat pump", 0), ("gas pipeline", 0),
]:
    if carrier not in network.carriers.index:
        network.add("Carrier", carrier, co2_emissions=co2)

# =============================================================================
# 5. ELECTRICITY GENERATORS AND BATTERY STORAGE
# =============================================================================
def add_country_generators(network, country):
    CF_on  = df_onshorewind[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ") for h in network.snapshots]]
    CF_off = df_offshorewind[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ") for h in network.snapshots]]
    CF_pv  = df_solar[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ") for h in network.snapshots]]

    network.add("Generator", f"Onshore wind {country}", bus=country, carrier="onshorewind",
                p_nom_extendable=True, capital_cost=annuity(27, 0.07) * 1118775,
                marginal_cost=0, p_max_pu=CF_on.values)
    network.add("Generator", f"Offshore wind {country}", bus=country, carrier="offshorewind",
                p_nom_extendable=True, capital_cost=annuity(27, 0.07) * 2115944,
                marginal_cost=0, p_max_pu=CF_off.values)
    network.add("Generator", f"Solar {country}", bus=country, carrier="solar",
                p_nom_extendable=True, capital_cost=annuity(25, 0.07) * 450000,
                marginal_cost=0, p_max_pu=CF_pv.values)
    network.add("Generator", f"OCGT {country}", bus=country, carrier="gas",
                p_nom_extendable=True, capital_cost=annuity(25, 0.07) * 453960,
                marginal_cost=gas_price / 0.41)
    network.add("Generator", f"CCGT {country}", bus=country, carrier="gas",
                p_nom_extendable=True, capital_cost=annuity(25, 0.07) * 880000,
                marginal_cost=gas_price / 0.56)
    network.add("StorageUnit", f"battery storage {country}", bus=country,
                carrier="battery storage", p_nom_extendable=True, max_hours=2,
                capital_cost=annuity(20, 0.07) * 2 * 288000,
                efficiency_store=0.98, efficiency_dispatch=0.97,
                cyclic_state_of_charge=True)

for c in countries:
    add_country_generators(network, c)

# =============================================================================
# 6. PUMPED HYDRO STORAGE
# =============================================================================
def load_hydro_inflow(csv_path, year=2012):
    df = pd.read_csv(csv_path)
    df = df[df["Year"] == year].copy()
    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    return df.set_index('date')['Inflow [GWh]'].reindex(naive_snapshots).fillna(0).values

for country, csv, p_nom_max in [
    ("SWE", 'data/inflow/Hydro_Inflow_SE.csv', 16000),
    ("NOR", 'data/inflow/Hydro_Inflow_NO.csv', 33000),
    ("DEU", 'data/inflow/Hydro_Inflow_DE.csv',  7000),
]:
    network.add("StorageUnit", f"pumped hydro {country}",
                bus=country, carrier="pumped hydro",
                p_nom_extendable=True, p_nom_max=p_nom_max, max_hours=8,
                capital_cost=annuity(80, 0.07) * 400000,
                efficiency_store=0.9, efficiency_dispatch=0.9,
                cyclic_state_of_charge=True, marginal_cost=1,
                inflow=load_hydro_inflow(csv))

# =============================================================================
# 7. HEAT PUMPS (electricity → heat, temperature-dependent COP)
# =============================================================================
country_temp_col = {"DNK": "DK", "SWE": "SE", "NOR": "NO", "DEU": "DE"}

def cop(t_source, t_sink=55):
    delta_t = t_sink - t_source
    return 6.81 - 0.121 * delta_t + 0.00063 * delta_t ** 2

for c in countries:
    temp_series = df_temp[country_temp_col[c]].reindex(naive_snapshots).ffill().bfill()
    cop_values = cop(temp_series).clip(lower=1.0).values

    # Heat pump: bus0=electricity, bus1=heat, efficiency=COP (temperature-dependent)
    # Capital cost: ~986 000 €/MW_e, 33-year lifetime (literature/DEA)
    network.add("Link", f"heat pump {c}",
                bus0=c, bus1=f"{c} heat", carrier="heat pump",
                p_nom_extendable=True,
                capital_cost=annuity(33, 0.07) * 986361,
                efficiency=cop_values)

# =============================================================================
# 8. GAS BOILERS (backup heat supply)
# DEA Technology Catalogue: large gas boiler, 20 yr, 62 000 €/MW_th, η=0.90
# =============================================================================
for c in countries:
    network.add("Generator", f"gas boiler {c}",
                bus=f"{c} heat", carrier="gas",
                p_nom_extendable=True,
                capital_cost=annuity(20, 0.07) * 62000,
                marginal_cost=gas_price / 0.9)

# =============================================================================
# 9. ELECTRICITY TRANSMISSION LINES
# =============================================================================
x_line = 0.1
for name, b0, b1, s_nom in [
    ("DK-NO", "DNK", "NOR", 1632),
    ("DK-SE", "DNK", "SWE", 2415),
    ("DK-DE", "DNK", "DEU", 3500),
    ("SE-NO", "SWE", "NOR", 3945),
    ("SE-DE", "SWE", "DEU",  615),
    ("NO-DE", "NOR", "DEU", 1400),
]:
    network.add("Line", name, bus0=b0, bus1=b1, x=x_line, s_nom=s_nom)

# =============================================================================
# 9a. GAS NETWORK
# Gas buses, national supply generators, and cross-border pipeline links.
# Norway:  124 bcm/yr ≈ 1300 TWh/yr ≈ 150 GW average capacity
# Germany: external imports supply beyond what arrives from Norway via pipelines
# Denmark: 73 000 TJ/yr ÷ 3600 ≈ 2315 MW average capacity
# =============================================================================
for c in countries:
    network.add("Bus", f"{c}_gas")

network.add("Generator", "Norway gas supply",
            bus="NOR_gas", carrier="gas",
            p_nom=150000,   # MW
            marginal_cost=gas_price)

network.add("Generator", "Germany external gas supply",
            bus="DEU_gas", carrier="gas",
            p_nom=54000,    # MW
            marginal_cost=gas_price)

network.add("Generator", "Denmark gas supply",
            bus="DNK_gas", carrier="gas",
            p_nom=2315,     # MW
            marginal_cost=gas_price)

gas_pipeline_efficiency = 1.0
for i, (bus0, bus1, p_nom) in enumerate([
    ("DNK_gas", "SWE_gas", 3960),
    ("SWE_gas", "DNK_gas", 3960),
    ("DNK_gas", "NOR_gas", 1400),
    ("NOR_gas", "DNK_gas", 1400),
    ("DNK_gas", "DEU_gas", 3500),
    ("DEU_gas", "DNK_gas", 3500),
    ("NOR_gas", "DEU_gas", 40800),
    ("DEU_gas", "NOR_gas", 40800),
]):
    network.add("Link", f"gas_pipeline_{i}",
                bus0=bus0, bus1=bus1, carrier="gas pipeline",
                p_nom=p_nom,
                efficiency=gas_pipeline_efficiency,
                marginal_cost=0.0)

# =============================================================================
# 10a. CO2 CONSTRAINT
# =============================================================================
co2_emissions_sum = 65462371.23885886  # from task d
network.add("GlobalConstraint",
            "co2_limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=65e6)  # tonnes CO₂

# =============================================================================
# 10b. OPTIMISATION
# =============================================================================
network.optimize(solver_name="gurobi", solver_options={"OutputFlag": 0, "LogToConsole": 0})

print(f"Total system cost: {network.objective / 1e6:.2f} M€/yr")

# =============================================================================
# 11. SAVE RESULTS
# =============================================================================
os.makedirs('plots_part_i', exist_ok=True)

ELEC_TECHS = ["Onshore wind", "Offshore wind", "Solar", "OCGT", "CCGT",
              "battery storage", "pumped hydro"]

# Electricity capacity table [GW] (excludes gas boilers which are on heat bus)
elec_gens = [c for c in network.generators.index if not c.startswith("gas boiler")]
elec_caps = pd.concat([
    network.generators.p_nom_opt[elec_gens],
    network.storage_units.p_nom_opt,
]) / 1000
cap_elec = (
    elec_caps.rename_axis("name").reset_index(name="capacity")
    .assign(
        country=lambda df: df["name"].str.split().str[-1],
        tech=lambda df: df["name"].str.replace(r"\s+(DNK|SWE|NOR|DEU)$", "", regex=True)
    )
    .pivot(index="country", columns="tech", values="capacity")
    .reindex(index=["DNK", "SWE", "NOR", "DEU"], columns=ELEC_TECHS)
    .fillna(0)
    .rename(index={"DNK": "Denmark", "SWE": "Sweden", "NOR": "Norway", "DEU": "Germany"})
)
cap_elec.to_csv('plots_part_i/cap_elec_table.csv')

# Heat capacity table [GW]
hp_caps = network.links.p_nom_opt[
    [c for c in network.links.index if c.startswith("heat pump ")]].copy() / 1000
hp_caps.index = hp_caps.index.str.replace(r"heat pump\s+(DNK|SWE|NOR|DEU)$", r"\1", regex=True)
boiler_caps = network.generators.p_nom_opt[
    [c for c in network.generators.index if c.startswith("gas boiler ")]].copy() / 1000
boiler_caps.index = boiler_caps.index.str.replace(
    r"gas boiler\s+(DNK|SWE|NOR|DEU)$", r"\1", regex=True)
cap_heat = pd.DataFrame({"heat pump": hp_caps, "gas boiler": boiler_caps}).reindex(
    ["DNK", "SWE", "NOR", "DEU"]).fillna(0)
cap_heat.rename(index={"DNK": "Denmark", "SWE": "Sweden",
                        "NOR": "Norway", "DEU": "Germany"}, inplace=True)
cap_heat.to_csv('plots_part_i/cap_heat_table.csv')

# Annual electricity generation mix [TWh]
gen_by_name = network.generators_t.p[elec_gens].sum() / 1e6
gen_by_name.index = gen_by_name.index.str.replace(r"\s+(DNK|SWE|NOR|DEU)$", "", regex=True)
gen_by_tech = gen_by_name.groupby(level=0).sum()
sto_by_name = network.storage_units_t.p.clip(lower=0).sum() / 1e6
sto_by_name.index = sto_by_name.index.str.replace(r"\s+(DNK|SWE|NOR|DEU)$", "", regex=True)
sto_by_tech = sto_by_name.groupby(level=0).sum()
generation_mix = pd.concat([gen_by_tech, sto_by_tech]).reindex(ELEC_TECHS).fillna(0)
generation_mix.to_csv('plots_part_i/generation_mix.csv')

# Annual heat supply mix [TWh] per country
hp_cols = [c for c in network.links.index if c.startswith("heat pump ")]
hp_heat = (-network.links_t.p1[hp_cols]).sum() / 1e6
hp_heat.index = hp_heat.index.str.replace(r"heat pump\s+(DNK|SWE|NOR|DEU)$", r"\1", regex=True)
boiler_cols = [c for c in network.generators.index if c.startswith("gas boiler ")]
boiler_heat = network.generators_t.p[boiler_cols].sum() / 1e6
boiler_heat.index = boiler_heat.index.str.replace(
    r"gas boiler\s+(DNK|SWE|NOR|DEU)$", r"\1", regex=True)
heat_supply = pd.DataFrame({"heat pump": hp_heat, "gas boiler": boiler_heat}).reindex(
    ["DNK", "SWE", "NOR", "DEU"]).fillna(0)
heat_supply.rename(index={"DNK": "Denmark", "SWE": "Sweden",
                           "NOR": "Norway", "DEU": "Germany"}, inplace=True)
heat_supply.to_csv('plots_part_i/heat_supply_mix.csv')

# Hourly electricity dispatch by tech [MW] (for duration curves)
gen_disp = network.generators_t.p[elec_gens].copy()
gen_disp.columns = gen_disp.columns.str.replace(r"\s+(DNK|SWE|NOR|DEU)$", "", regex=True)
gen_t = gen_disp.T.groupby(level=0).sum().T
sto_disp = network.storage_units_t.p.copy()
sto_disp.columns = sto_disp.columns.str.replace(
    r"(battery storage|pumped hydro)\s+(DNK|SWE|NOR|DEU)$", r"\1", regex=True)
sto_t = sto_disp.T.groupby(level=0).sum().T
dispatch_by_tech = pd.concat([gen_t, sto_t], axis=1).T.groupby(level=0).sum().T
dispatch_by_tech.to_csv('plots_part_i/dispatch_by_tech.csv')

# DNK heat dispatch time series [MW]
hp_dnk = -network.links_t.p1.get("heat pump DNK", pd.Series(0, index=network.snapshots))
boiler_dnk = network.generators_t.p.get("gas boiler DNK", pd.Series(0, index=network.snapshots))
pd.DataFrame({"heat pump": hp_dnk, "gas boiler": boiler_dnk}).to_csv(
    'plots_part_i/heat_dispatch_dnk.csv')

# System cost
pd.Series({"system_cost_M_eur": network.objective / 1e6}).to_csv('plots_part_i/co2_results.csv')

print("Saved CSVs to plots_part_i/")
print("\nElectricity capacity [GW]:")
print(cap_elec)
print("\nHeat capacity [GW]:")
print(cap_heat)
print("\nAnnual generation mix [TWh]:")
print(generation_mix)
print("\nAnnual heat supply [TWh]:")
print(heat_supply)
