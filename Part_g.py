import pandas as pd
import pypsa
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATA
# -----------------------------
df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)  # in MWh
df_elec.index = pd.to_datetime(df_elec.index)

df_onshorewind = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0)
df_onshorewind.index = pd.to_datetime(df_onshorewind.index)

df_offshorewind = pd.read_csv('data/offshore_wind_1979-2017.csv', sep=';', index_col=0)
df_offshorewind.index = pd.to_datetime(df_offshorewind.index)

df_solar = pd.read_csv('data/pv_optimal.csv', sep=';', index_col=0)
df_solar.index = pd.to_datetime(df_solar.index)

# -----------------------------

# INITIALIZE NETWORK
# -----------------------------
network = pypsa.Network()
hours_in_2015 = pd.date_range('2015-01-01 00:00Z', '2015-12-31 23:00Z', freq='h')
network.set_snapshots(hours_in_2015.values)

countries = ["DNK", "SWE", "NOR", "DEU"]

for c in countries:
    network.add("Bus", c, v_nom=400)

for c in countries:
    network.add("Load", f"load_{c}", bus=c, p_set=df_elec[c].values)

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def annuity(n, r):
    """Calculate the annuity factor for an asset with lifetime n years and discount rate r."""
    if r > 0:
        return r / (1. - 1. / (1. + r) ** n)
    else:
        return 1 / n

# -----------------------------
# PART G - GAS NETWORK SETTINGS
# -----------------------------
gas_price = 30.0  # €/MWh_th
gas_pipeline_efficiency = 1.0

# Add carriers
for carrier in ["gas", "onshorewind", "offshorewind", "solar", "battery storage", "pumped hydro", "gas fuel", "gas pipeline"]:
    if carrier not in network.carriers.index:
        if carrier == "gas":
            network.add("Carrier", carrier, co2_emissions=0.19)
        else:
            network.add("Carrier", carrier)

# Add one gas bus per country
for c in countries:
    network.add("Bus", f"{c}_gas")


# Norway gas production based on real data 
# 124 bcm ≈ 1300 TWh/year ≈ 150 GW average capacity

network.add(
    "Generator",
    "Norway gas supply",
    bus="NOR_gas",
    carrier="gas",
    p_nom=150000,   # MW (≈150 GW)
    marginal_cost=gas_price # €/MWh_th
)
# Germany gas supply 
# 137000 TJ/year ÷ 3600 = 38.1 TWh/year → (38.1e6 MWh / 8760 h) ≈ 4350 MW

 # But
# Extra gas supply for Germany from outside the 4-country model
# Germany demand is higher than what can come from Norway,
# so the remaining gas is added as external supply at DEU_gas

network.add(
    "Generator",
    "Germany external gas supply",
    bus="DEU_gas",
    carrier="gas",
    p_nom=54000,   # MW
    marginal_cost=gas_price
)

# Denmark gas supply 
# 73000 TJ/year ÷ 3600 = 20.3 TWh/year → (20.3e6 MWh / 8760 h) ≈ 2315 MW
network.add(
    "Generator",
    "Denmark gas supply",
    bus="DNK_gas",
    carrier="gas",
    p_nom=2315,
    marginal_cost=gas_price
)

# Fixed gas pipeline capacities in MW
gas_links = [
    ("DNK_gas", "SWE_gas", 1958),
    ("SWE_gas", "DNK_gas", 1958),
    ("DNK_gas", "NOR_gas", 1400),
    ("NOR_gas", "DNK_gas", 1400),
    ("DNK_gas", "DEU_gas", 3500),
    ("DEU_gas", "DNK_gas", 3500),
    ("NOR_gas", "DEU_gas", 40800),
    ("DEU_gas", "NOR_gas", 40800),
]

for i, (bus0, bus1, p_nom) in enumerate(gas_links):
    network.add(
        "Link",
        f"gas_pipeline_{i}",
        bus0=bus0,
        bus1=bus1,
        carrier="gas pipeline",
        p_nom=p_nom,
        efficiency=gas_pipeline_efficiency,
        marginal_cost=0.0
    )

# -----------------------------
# ADD COUNTRY GENERATORS
# -----------------------------
def add_country_generators(network, country, df_onshorewind, df_offshorewind, df_solar):

    # Capacity factors aligned to snapshots
    CF_on = df_onshorewind[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
    CF_off = df_offshorewind[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
    CF_pv = df_solar[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]

    # Costs
    capital_cost_onshorewind = annuity(27, 0.07) * 1118775
    capital_cost_offshorewind = annuity(27, 0.07) * 2115944
    capital_cost_solar = annuity(25, 0.07) * 450000
    capital_cost_OCGT = annuity(25, 0.07) * 453960
    capital_cost_CCGT = annuity(25, 0.07) * 880000

    efficiency_OCGT = 0.41
    efficiency_CCGT = 0.56
    
    # PART G FIX: convert cost from MW_e to MW_th
    capital_cost_OCGT_link = capital_cost_OCGT * efficiency_OCGT
    capital_cost_CCGT_link = capital_cost_CCGT * efficiency_CCGT
    
    # PART G: gas fuel is modeled explicitly, so only optional variable O&M remains here
    variable_om_OCGT = 0.0
    variable_om_CCGT = 0.0

    network.add(
        "Generator", f"Onshore wind {country}",
        bus=country, carrier="onshorewind",
        p_nom_extendable=True,
        capital_cost=capital_cost_onshorewind,
        marginal_cost=0,
        p_max_pu=CF_on.values
    )

    network.add(
        "Generator", f"Offshore wind {country}",
        bus=country, carrier="offshorewind",
        p_nom_extendable=True,
        capital_cost=capital_cost_offshorewind,
        marginal_cost=0,
        p_max_pu=CF_off.values
    )

    network.add(
        "Generator", f"Solar {country}",
        bus=country, carrier="solar",
        p_nom_extendable=True,
        capital_cost=capital_cost_solar,
        marginal_cost=0,
        p_max_pu=CF_pv.values
    )

    # PART G: gas plants are modeled as gas-to-power links
    network.add(
        "Link", f"OCGT {country}",
        bus0=f"{country}_gas",
        bus1=country,
        carrier="gas",
        p_nom_extendable=True,
        capital_cost=capital_cost_OCGT_link,
        marginal_cost=variable_om_OCGT,
        efficiency=efficiency_OCGT
    )

    network.add(
        "Link", f"CCGT {country}",
        bus0=f"{country}_gas",
        bus1=country,
        carrier="gas",
        p_nom_extendable=True,
        capital_cost=capital_cost_CCGT_link,
        marginal_cost=variable_om_CCGT,
        efficiency=efficiency_CCGT
    )

    network.add(
        "StorageUnit", f"battery storage {country}",
        bus=country, carrier="battery storage",
        p_nom_extendable=True,
        p_nom_max=10000,
        max_hours=2,
        capital_cost=annuity(20, 0.07) * 2 * 288000,
        efficiency_store=0.98,
        efficiency_dispatch=0.97,
        cyclic_state_of_charge=True
    )

for c in countries:
    add_country_generators(network, c, df_onshorewind, df_offshorewind, df_solar)

# -----------------------------
# HYDRO / PUMPED HYDRO
# -----------------------------
test_SWE = pd.read_csv('data/inflow/Hydro_Inflow_SE.csv')
test_SWE = test_SWE[test_SWE["Year"] == 2012]
test_SWE['date'] = pd.to_datetime(test_SWE[['Year', 'Month', 'Day']])
hydro_inflow_SWE = test_SWE.set_index('date')['Inflow [GWh]'].reindex(network.snapshots).fillna(0).values

test_NOR = pd.read_csv('data/inflow/Hydro_Inflow_NO.csv')
test_NOR = test_NOR[test_NOR["Year"] == 2012]
test_NOR['date'] = pd.to_datetime(test_NOR[['Year', 'Month', 'Day']])
hydro_inflow_NOR = test_NOR.set_index('date')['Inflow [GWh]'].reindex(network.snapshots).fillna(0).values

test_DEU = pd.read_csv('data/inflow/Hydro_Inflow_DE.csv')
test_DEU = test_DEU[test_DEU["Year"] == 2012]
test_DEU['date'] = pd.to_datetime(test_DEU[['Year', 'Month', 'Day']])
hydro_inflow_DEU = test_DEU.set_index('date')['Inflow [GWh]'].reindex(network.snapshots).fillna(0).values

network.add(
    "StorageUnit", "pumped hydro SWE",
    bus="SWE",
    carrier="pumped hydro",
    p_nom_extendable=True,
    p_nom_max=16000,
    max_hours=8,
    capital_cost=annuity(80, 0.07) * 400000,
    efficiency_store=0.9,
    efficiency_dispatch=0.9,
    cyclic_state_of_charge=True,
    inflow=hydro_inflow_SWE,
    marginal_cost=1
)

network.add(
    "StorageUnit", "pumped hydro NOR",
    bus="NOR",
    carrier="pumped hydro",
    p_nom_extendable=True,
    p_nom_max=33000,
    max_hours=8,
    capital_cost=annuity(80, 0.07) * 400000,
    efficiency_store=0.9,
    efficiency_dispatch=0.9,
    cyclic_state_of_charge=True,
    inflow=hydro_inflow_NOR,
    marginal_cost=1
)

network.add(
    "StorageUnit", "pumped hydro DEU",
    bus="DEU",
    carrier="pumped hydro",
    p_nom_extendable=True,
    p_nom_max=7000,
    max_hours=8,
    capital_cost=annuity(80, 0.07) * 400000,
    efficiency_store=0.9,
    efficiency_dispatch=0.9,
    cyclic_state_of_charge=True,
    inflow=hydro_inflow_DEU,
    marginal_cost=1
)

# -----------------------------
# ELECTRICITY TRANSMISSION LINES
# -----------------------------
x_line = 0.1
network.add("Line", "DK-NO", bus0="DNK", bus1="NOR", x=x_line, s_nom=1632)
network.add("Line", "DK-SE", bus0="DNK", bus1="SWE", x=x_line, s_nom=2415)
network.add("Line", "DK-DE", bus0="DNK", bus1="DEU", x=x_line, s_nom=3500)
network.add("Line", "SE-NO", bus0="SWE", bus1="NOR", x=x_line, s_nom=3945)
network.add("Line", "SE-DE", bus0="SWE", bus1="DEU", x=x_line, s_nom=615)
network.add("Line", "NO-DE", bus0="NOR", bus1="DEU", x=x_line, s_nom=1400)



# -----------------------------
# OPTIMIZATION
# -----------------------------
network.optimize(solver_name="gurobi", solver_options={"OutputFlag": 0, "LogToConsole": 0})

# -----------------------------
# CAPACITY RESULTS
# -----------------------------
print("Optimal generator capacities [GW]:")
print(network.generators.p_nom_opt.div(1e3))

print("\nOptimal link capacities [GW]:")
print(network.links.p_nom_opt.div(1e3))

optimal_capacities = pd.concat([
    network.generators.p_nom_opt,
    network.storage_units.p_nom_opt,
    network.links.p_nom_opt
]) / 1000

cap_table = (
    optimal_capacities.rename_axis("name").reset_index(name="capacity")
    .assign(
        country=lambda df: df["name"].str.split().str[-1],
        tech=lambda df: df["name"].str.replace(r"\s+(DNK|SWE|NOR|DEU)$", "", regex=True)
    )
    .pivot(index="country", columns="tech", values="capacity")
    .reindex(
        index=["DNK", "SWE", "NOR", "DEU"],
        columns=["Onshore wind", "Offshore wind", "Solar", "OCGT", "CCGT", "battery storage", "pumped hydro"]
    )
    .fillna(0)
)

cap_table = cap_table.rename(index={
    "DNK": "Denmark",
    "SWE": "Sweden",
    "NOR": "Norway",
    "DEU": "Germany"
})

fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
cap_table.plot(
    kind="bar",
    stacked=True,
    color=['blue', 'dodgerblue', 'orange', 'crimson', 'darkviolet', 'lightgreen', 'pink'],
    rot=0,
    ax=ax
)
ax.set_ylabel("Installed capacity [GW]")
ax.set_xlabel("")
ax.legend(['Onshore wind', 'Offshore wind', 'Solar PV', 'Gas (OCGT)', 'Gas (CCGT)', 'Battery storage', 'Hydro'])
plt.tight_layout()
plt.savefig("1g_capacities.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# TOTAL INSTALLED CAPACITY PIE CHART
# -----------------------------
total_capacities = cap_table.sum(axis=0)
tech_labels = ['Onshore wind', 'Offshore wind', 'Solar PV', 'Gas (OCGT)', 'Gas (CCGT)', 'Battery storage', 'Hydro']
perc = 100 * total_capacities / total_capacities.sum()
labels = [f"{tech}\n{p:.1f}%" for tech, p in zip(tech_labels, perc)]

fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
ax.pie(
    total_capacities,
    labels=labels,
    colors=['blue', 'dodgerblue', 'orange', 'crimson', 'darkviolet', 'lightgreen', 'pink'],
    startangle=90,
    labeldistance=1.18
)
plt.tight_layout()
plt.savefig("1g_total_capacities.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# ANNUAL GENERATION MIX
# PART G: gas generation now comes from links, not generators
# -----------------------------
gen_by_name = network.generators_t.p.sum() / 1e6
gen_by_name.index = gen_by_name.index.str.replace(r"\s+(DNK|SWE|NOR|DEU)$", "", regex=True)
gen_by_tech = gen_by_name.groupby(gen_by_name.index).sum()

storage_by_name = network.storage_units_t.p.clip(lower=0).sum() / 1e6
storage_by_name.index = storage_by_name.index.str.replace(r"\s+(DNK|SWE|NOR|DEU)$", "", regex=True)
storage_by_tech = storage_by_name.groupby(storage_by_name.index).sum()

# PART G: gas-to-power output from links
gas_link_output = network.links_t.p1[[name for name in network.links.index if name.startswith("OCGT ") or name.startswith("CCGT ")]].sum() / 1e6
gas_link_output.index = gas_link_output.index.str.replace(r"\s+(DNK|SWE|NOR|DEU)$", "", regex=True)
gas_link_output = gas_link_output.groupby(gas_link_output.index).sum()

generation_mix = pd.concat([gen_by_tech, gas_link_output, storage_by_tech]).reindex(
    ["Onshore wind", "Offshore wind", "Solar", "OCGT", "CCGT", "battery storage", "pumped hydro"]
).fillna(0)

print("Annual generation mix in TWh:")
print(generation_mix)

perc_gen = 100 * generation_mix / generation_mix.sum()
gen_labels = [f"{tech}\n{p:.1f}%" for tech, p in zip(tech_labels, perc_gen)]

fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
ax.pie(
    generation_mix,
    labels=gen_labels,
    colors=['blue', 'dodgerblue', 'orange', 'crimson', 'darkviolet', 'lightgreen', 'pink'],
    startangle=90,
    labeldistance=1.18,
    wedgeprops={'linewidth': 0}
)
plt.title('Annual electricity generation mix', y=1.05, fontweight='bold')
plt.tight_layout()
plt.savefig('1g_annual_generation_mix.png', dpi=300)
plt.show()

# -----------------------------
# DURATION CURVE
# PART G: include gas generation from links
# -----------------------------
gen_dispatch = network.generators_t.p.copy()
gen_dispatch.columns = gen_dispatch.columns.str.replace(r"\s+(DNK|SWE|NOR|DEU)$", "", regex=True)
gen_by_tech_time = gen_dispatch.T.groupby(level=0).sum().T

storage_dispatch = network.storage_units_t.p.copy()
storage_dispatch.columns = storage_dispatch.columns.str.replace(r"(battery storage|pumped hydro)\s+(DNK|SWE|NOR|DEU)$", r"\1", regex=True)
storage_by_tech_time = storage_dispatch.T.groupby(level=0).sum().T

gas_dispatch = network.links_t.p1[[name for name in network.links.index if name.startswith("OCGT ") or name.startswith("CCGT ")]].copy()
gas_dispatch.columns = gas_dispatch.columns.str.replace(r"\s+(DNK|SWE|NOR|DEU)$", "", regex=True)
gas_by_tech_time = gas_dispatch.T.groupby(level=0).sum().T

dispatch_by_tech_time = pd.concat([gen_by_tech_time, gas_by_tech_time, storage_by_tech_time], axis=1)
dispatch_by_tech_time = dispatch_by_tech_time.T.groupby(level=0).sum().T

tech_order = ["Onshore wind", "Offshore wind", "Solar", "OCGT", "CCGT", "battery storage", "pumped hydro"]
plot_labels = ["Onshore wind", "Offshore wind", "Solar PV", "Gas (OCGT)", "Gas (CCGT)", "Battery storage", "Hydro"]
plot_colors = ['blue', 'dodgerblue', 'orange', 'crimson', 'darkviolet', 'lightgreen', 'pink']

plt.figure(figsize=(8, 5), dpi=300)
for tech, label, color in zip(tech_order, plot_labels, plot_colors):
    if tech in dispatch_by_tech_time.columns:
        sorted_dispatch = dispatch_by_tech_time[tech].sort_values(ascending=False).reset_index(drop=True)
        plt.plot(sorted_dispatch, label=label, color=color, lw=2.5)
plt.axhline(0, color='black', lw=0.8)
plt.ylabel('Dispatch [MWh/h]')
plt.xlabel('Hours')
plt.title('Duration curve of generation and storage operation')
plt.legend(fancybox=True, shadow=True, loc='best')
plt.tight_layout()
plt.savefig('1g_duration_curve.png', dpi=300)
plt.show()

# -----------------------------
# PART G - GAS FLOW ANALYSIS
# -----------------------------
gas_link_names = [name for name in network.links.index if name.startswith("gas_pipeline_")]
gas_flows = network.links_t.p0[gas_link_names].copy()

total_gas_transport_twh = gas_flows.abs().sum().sum() / 1e6
total_electricity_transport_twh = network.lines_t.p0.abs().sum().sum() / 1e6

print(f"Total electricity transported: {total_electricity_transport_twh:.2f} TWh")
print(f"Total gas transported: {total_gas_transport_twh:.2f} TWh")

# -----------------------------
# PART G - GAS TO POWER OUTPUT BY LINK
# -----------------------------
gas_power_links = [name for name in network.links.index if name.startswith("OCGT ") or name.startswith("CCGT ")]
gas_power_output = network.links_t.p1[gas_power_links].sum() / 1e6

print("Annual gas-to-power generation by plant [TWh]:")
print(gas_power_output)

# -----------------------------
# PART G - COMPARISON GRAPH
# Electricity network vs gas network
# -----------------------------
comparison = pd.Series({
    "Electricity network": total_electricity_transport_twh,
    "Gas network": total_gas_transport_twh
})

fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
comparison.plot(kind="bar", ax=ax, color=["steelblue", "darkorange"], rot=0)
ax.set_ylabel("Transported energy [TWh]")
ax.set_title("Comparison of transported energy in electricity and gas networks")
plt.tight_layout()
plt.savefig("1g_transport_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# PART G - GAS PIPELINE FLOW GRAPH
# -----------------------------
avg_abs_gas_flow = gas_flows.abs().mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
avg_abs_gas_flow.plot(kind="bar", ax=ax, color="darkorange")
ax.set_ylabel("Average absolute gas flow [MW]")
ax.set_title("Average absolute flow in gas pipelines")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("1g_gas_pipeline_flows.png", dpi=300, bbox_inches="tight")
plt.show()