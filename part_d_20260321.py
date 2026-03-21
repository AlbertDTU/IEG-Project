import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# LOAD DATA
# -------------------------
df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)
df_elec.index = pd.to_datetime(df_elec.index)

print("Electricity demand head:\n", df_elec.head())


df_onshorewind = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0)
df_onshorewind.index = pd.to_datetime(df_onshorewind.index)

df_offshorewind = pd.read_csv('data/offshore_wind_1979-2017.csv', sep=';', index_col=0)
df_offshorewind.index = pd.to_datetime(df_offshorewind.index)

df_solar = pd.read_csv('data/pv_optimal.csv', sep=';', index_col=0)
df_solar.index = pd.to_datetime(df_solar.index, utc=True)  

country = 'DNK'

# -------------------------
# NETWORK
# -------------------------
network = pypsa.Network()

# snapshots: create UTC-aware first, then convert to naive for PyPSA
hours_utc = pd.date_range('2015-01-01 00:00','2015-12-31 23:00', freq='h', tz='UTC')
hours_naive = hours_utc.tz_convert(None)  # PyPSA requires naive timestamps
network.set_snapshots(hours_naive)

# Carriers
for c in ["gas","onshorewind","offshorewind","solar","hydro"]:
    network.add("Carrier",c)

# Buses
for c in ["Denmark","Sweden","Norway","Germany"]:
    network.add("Bus",f"bus {c}",v_nom=400)

# Lines
# The line capacities are based on Figure 8 page 156 https://www.agora-energiewende.org/publications/increased-integration-of-the-nordic-and-german-electricity-systems-full-report?utm_source=chatgpt.com#downloads
network.add("Line","DK-NO",bus0="bus Denmark",bus1="bus Norway",x=0.1,r=0.001,s_nom=1000)
network.add("Line","DK-DE",bus0="bus Denmark",bus1="bus Germany",x=0.1,r=0.001,s_nom=600)
network.add("Line","DK-SE",bus0="bus Denmark",bus1="bus Sweden",x=0.1,r=0.001,s_nom=700)
# network.add("Line","SE-NO",bus0="bus Sweden",bus1="bus Norway",x=0.1,r=0.001,s_nom=3945)
network.add("Line","SE-DE",bus0="bus Sweden",bus1="bus Germany",x=0.1,r=0.001,s_nom=600)
# network.add("Line","NO-DE",bus0="bus Norway",bus1="bus Germany",x=0.1,r=0.001,s_nom=1400)

# -------------------------
# LOADS
# -------------------------
network.add("Load","Denmark load",bus="bus Denmark",
            p_set=df_elec[country].values.flatten())

# network.add("Load","Germany load",bus="bus Germany",p_set=522260)
network.add("Load","Germany load",bus="bus Germany",
            p_set=df_elec["DEU"].reindex(hours_naive).fillna(0).values)
# network.add("Load","Sweden load",bus="bus Sweden",p_set=138710)
network.add("Load","Sweden load",bus="bus Sweden",
            p_set=df_elec["SWE"].reindex(hours_naive).fillna(0).values)
# network.add("Load","Norway load",bus="bus Norway",p_set=136700)
network.add("Load","Norway load",bus="bus Norway",
            p_set=df_elec["NOR"].reindex(hours_naive).fillna(0  ).values)

print("\nAnnual load (TWh):")

print("Denmark:", df_elec["DNK"].sum() / 1e6)
print("Germany:", df_elec["DEU"].sum() / 1e6)
print("Sweden:", df_elec["SWE"].sum() / 1e6)
print("Norway:", df_elec["NOR"].sum() / 1e6)
# -------------------------
# HELPER
# -------------------------
def annuity(n,r):
    return r/(1.-1./(1.+r)**n) if r>0 else 1/n

# -------------------------
# CAPACITY FACTORS
# -------------------------
# Use UTC-aware snapshots to match CSV, then fill missing hours with 0
CF_wind = df_onshorewind[country].reindex(hours_utc, fill_value=0)
CF_off = df_offshorewind[country].reindex(hours_utc, fill_value=0)


CF_solar_de = df_solar["DEU"].reindex(hours_utc, fill_value=0)
print("Solar DEU CF head:\n", CF_solar_de.head(20))
print("Germany solar CF min/max:", CF_solar_de.min(), CF_solar_de.max())
CF_solar_dk = df_solar["DNK"].reindex(hours_utc, fill_value=0)

CF_onshorewind_de = df_onshorewind["DEU"].reindex(hours_utc, fill_value=0)
CF_onshorewind_swe = df_onshorewind["SWE"].reindex(hours_utc, fill_value=0)

CF_offshorewind_de = df_offshorewind["DEU"].reindex(hours_utc, fill_value=0)
CF_offshorewind_swe = df_offshorewind["SWE"].reindex(hours_utc, fill_value=0)
# -------------------------
# DENMARK GENERATORS
# -------------------------
network.add("Generator","Onshore wind Denmark",bus="bus Denmark",
            p_nom_extendable=True,
            capital_cost=annuity(27,0.07)*1118775,
            marginal_cost=0,p_max_pu=CF_wind.values)

network.add("Generator","Offshore wind Denmark",bus="bus Denmark",
            p_nom_extendable=True,
            capital_cost=annuity(27,0.07)*2115944,
            marginal_cost=0,p_max_pu=CF_off.values)

network.add("Generator","Solar Denmark",bus="bus Denmark",
            p_nom_extendable=True,
            capital_cost=annuity(25,0.07)*450000,
            marginal_cost=0,p_max_pu=CF_solar_dk.values)

network.add("Generator","OCGT Denmark",bus="bus Denmark",
            p_nom_extendable=True,
            capital_cost=annuity(25,0.07)*453960,
            marginal_cost=30/0.41)

network.add("Generator","CCGT Denmark",bus="bus Denmark",
            p_nom_extendable=True,
            capital_cost=annuity(25,0.07)*880000,
            marginal_cost=30/0.56)

# -------------------------
# NEIGHBOUR GENERATION
# -------------------------
# GERMANY
network.add("Generator","Solar Germany",bus="bus Germany",
            p_nom_extendable=True,
            capital_cost=annuity(25,0.07)*450000,
            p_max_pu=CF_solar_de.values,
            marginal_cost=0)

network.add("Generator","Onshore wind Germany",bus="bus Germany",
            p_nom_extendable=True,
            capital_cost=annuity(27,0.07)*1118775,
            p_max_pu=CF_onshorewind_de.values,
            marginal_cost=0)

network.add("Generator","Offshore wind Germany",bus="bus Germany",
            p_nom_extendable=True,
            capital_cost=annuity(27,0.07)*2115944,
            p_max_pu=CF_offshorewind_de.values,
            marginal_cost=0)

# SWEDEN
network.add("Generator","Onshore wind Sweden",bus="bus Sweden",
            p_nom_extendable=True,
            capital_cost=annuity(27,0.07)*1118775,
            p_max_pu=CF_onshorewind_swe.values,
            marginal_cost=0)

network.add("Generator","Offshore wind Sweden",bus="bus Sweden",
            p_nom_extendable=True,
            capital_cost=annuity(27,0.07)*2115944,
            p_max_pu=CF_offshorewind_swe.values,
            marginal_cost=0)
# NORWAY
network.add("Generator","Gas Norway",bus="bus Norway",
            p_nom_extendable=True,
            marginal_cost=30/0.56)

# -------------------------
# HYDRO 
# -------------------------
# hydro Sweden
_df_hydro = pd.read_csv('data/inflow/Hydro_Inflow_SE.csv')
_df_hydro['date'] = pd.to_datetime(_df_hydro[['Year','Month','Day']])
_df_hydro['doy'] = _df_hydro['date'].dt.dayofyear

_avg_inflow = _df_hydro.groupby('doy')['Inflow [GWh]'].mean()

_inflow_gwh = pd.Series(
    [_avg_inflow.loc[ts.dayofyear] for ts in hours_naive],
    index=hours_naive
)
_inflow_mw_swe = _inflow_gwh * 1000 / 24
network.add(
    "StorageUnit",
    "Hydro Sweden",
    bus="bus Sweden",
    carrier="hydro",
    p_nom=16510,
    max_hours=2,
    inflow=_inflow_mw_swe,
    efficiency = 0.8, # https://energystorageeurope.eu/wp-content/uploads/2016/07/EASE_TD_Mechanical_PHS.pdf
    marginal_cost=20 # increased marginal cost to reflect higher opportunity cost of water in Sweden, which has more hydro resources
)
# hydro Norway


_df_hydro = pd.read_csv('data/inflow/Hydro_Inflow_NO.csv')
_df_hydro['date'] = pd.to_datetime(_df_hydro[['Year','Month','Day']])
_df_hydro['doy'] = _df_hydro['date'].dt.dayofyear

_avg_inflow = _df_hydro.groupby('doy')['Inflow [GWh]'].mean()

_inflow_gwh = pd.Series(
    [_avg_inflow.loc[ts.dayofyear] for ts in hours_naive],
    index=hours_naive
)

_inflow_mw_nor = _inflow_gwh * 1000 / 24

network.add(
    "StorageUnit",
    "Hydro Norway",
    bus="bus Norway",
    carrier="hydro",
    p_nom=34700,
    max_hours=2,
    inflow=_inflow_mw_nor,
    efficiency = 0.8, # https://energystorageeurope.eu/wp-content/uploads/2016/07/EASE_TD_Mechanical_PHS.pdf
    marginal_cost=20 # increased marginal cost to reflect higher opportunity cost of water in Norway, which has more hydro resources
)

# -------------------------
# OPTIMIZATION
# -------------------------
network.optimize(solver_name='highs')

network.storage_units_t.p_dispatch
print("\nHydro generation (TWh):")
print(network.storage_units_t.p_dispatch.sum() / 1e6)
print(network.storage_units_t.p_dispatch.describe())

print("Status:", network.model.status)
print("Termination:", network.model.termination_condition)

# -------------------------
# RESULTS
# -------------------------
if network.model.status == "ok":

    print("\nInstalled capacities (GW):")
    print(network.generators.p_nom_opt.div(1e3))

    print("\nTotal generation (TWh):")
    print(network.generators_t.p.sum().div(1e6))



else:
    print("Optimization failed.")

################################################
# PLOTS
################################################
# -------------------------
# INSTALLED CAPACITY MIX (INCLUDING HYDRO)
# -------------------------

# -------------------------
# INSTALLED CAPACITY MIX (INCLUDING HYDRO)
# -------------------------

# Generator capacities
gen_caps = network.generators.p_nom_opt.copy()

# Hydro capacities (from storage units)
hydro_caps = network.storage_units.p_nom.copy()

# Combine
all_caps = pd.concat([gen_caps, hydro_caps])

# Aggregate technologies
tech_map = {
    "Onshore wind Denmark": "Onshore wind",
    "Onshore wind Germany": "Onshore wind",
    "Onshore wind Sweden": "Onshore wind",
    
    "Offshore wind Denmark": "Offshore wind",
    "Offshore wind Germany": "Offshore wind",
    "Offshore wind Sweden": "Offshore wind",
    
    "Solar Denmark": "Solar",
    "Solar Germany": "Solar",
    
    "OCGT Denmark": "Gas",
    "CCGT Denmark": "Gas",
    "Gas Norway": "Gas",
    
    "Hydro Sweden": "Hydro",
    "Hydro Norway": "Hydro"
}

# Aggregate by technology
cap_by_tech = all_caps.groupby(tech_map).sum()

# Convert to GW
cap_by_tech = cap_by_tech / 1e3

# List of colors in the order you want (must match number of aggregated techs)
colors_list = ['blue', 'dodgerblue', 'orange', 'crimson', 'darkviolet']  
# Corresponds to: Onshore wind, Offshore wind, Solar, Gas, Hydro

# Map each technology to a color
tech_labels = cap_by_tech.index.tolist()
color_map = dict(zip(tech_labels, colors_list))
plot_colors = [color_map[label] for label in tech_labels]

# Plot
plt.figure(figsize=(6,5), dpi=300)
plt.pie(
    cap_by_tech.values,
    colors=plot_colors,
    labels=[f"{l}\n{s:.1f} GW" for l, s in zip(tech_labels, cap_by_tech.values)],
    wedgeprops={'linewidth':0}
)
plt.axis('equal')
plt.title('Installed capacity mix (optimal)', fontweight='bold')
plt.tight_layout()
plt.savefig('1c_installed_capacity_mix.png', dpi=300)
plt.show()

# -------------------------
# ANNUAL GENERATION MIX BY TECHNOLOGY
# -------------------------

# Sum annual generation for generators
gen_dispatch = network.generators_t.p.sum(axis=0)  # total per generator
hydro_dispatch = network.storage_units_t.p_dispatch.sum(axis=0)  # total per storage unit

# Combine
all_dispatch = pd.concat([gen_dispatch, hydro_dispatch])

# Map generator/storage names to technology
tech_map_dispatch = {
    "Onshore wind Denmark": "Onshore wind",
    "Onshore wind Germany": "Onshore wind",
    "Onshore wind Sweden": "Onshore wind",
    
    "Offshore wind Denmark": "Offshore wind",
    "Offshore wind Germany": "Offshore wind",
    "Offshore wind Sweden": "Offshore wind",
    
    "Solar Denmark": "Solar",
    "Solar Germany": "Solar",
    
    "OCGT Denmark": "Gas",
    "CCGT Denmark": "Gas",
    "Gas Norway": "Gas",
    
    "Hydro Sweden": "Hydro",
    "Hydro Norway": "Hydro"
}

gen_by_tech = all_dispatch.groupby(tech_map_dispatch).sum() / 1e6  # convert to TWh

# Colors (reuse your list)
colors_list = ['blue', 'dodgerblue', 'orange', 'crimson', 'darkviolet']  
tech_labels = gen_by_tech.index.tolist()
color_map = dict(zip(tech_labels, colors_list))
plot_colors = [color_map[label] for label in tech_labels]

# Plot annual generation pie chart
plt.figure(figsize=(6,5), dpi=300)
plt.pie(
    gen_by_tech.values,
    colors=plot_colors,
    labels=[f"{l}\n{s:.1f} TWh" for l, s in zip(tech_labels, gen_by_tech.values)],
    wedgeprops={'linewidth':0}
)
plt.axis('equal')
plt.title('Annual Generation Mix by Technology', fontweight='bold')
plt.tight_layout()
plt.savefig('annual_generation_mix.png', dpi=300)
plt.show()

# -------------------------
# GENERATION DURATION CURVE BY TECHNOLOGY
# -------------------------

# Group dispatch by technology per hour
dispatch = network.generators_t.p.copy()
dispatch["Hydro"] = network.storage_units_t.p_dispatch.sum(axis=1)

# Group by tech
dispatch_grouped = pd.DataFrame({
    "Onshore wind": dispatch.filter(like="Onshore wind").sum(axis=1),
    "Offshore wind": dispatch.filter(like="Offshore wind").sum(axis=1),
    "Solar": dispatch.filter(like="Solar").sum(axis=1),
    "Gas": dispatch.filter(like="OCGT").sum(axis=1) + dispatch.filter(like="CCGT").sum(axis=1) + dispatch.filter(like="Gas").sum(axis=1),
    "Hydro": dispatch["Hydro"]
})

# Plot duration curves for each technology
plt.figure(figsize=(7,5), dpi=300)
for tech in dispatch_grouped.columns:
    sorted_curve = dispatch_grouped[tech].sort_values(ascending=False).values / 1e3  # convert MW -> GW
    plt.plot(sorted_curve, label=tech, color=color_map[tech])

plt.xlabel("Hours")
plt.ylabel("Generation (GW)")
plt.title("Generation Duration Curve by Technology", fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("duration_curve_by_tech.png", dpi=300)
plt.show()