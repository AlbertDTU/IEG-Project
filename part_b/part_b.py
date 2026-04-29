import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0) # in MWh
df_elec.index = pd.to_datetime(df_elec.index) #change index to datatime

df_onshorewind = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0)
df_onshorewind.index = pd.to_datetime(df_onshorewind.index)

df_offshorewind = pd.read_csv('data/offshore_wind_1979-2017.csv', sep=';', index_col=0)
df_offshorewind.index = pd.to_datetime(df_offshorewind.index)

df_solar = pd.read_csv('data/pv_optimal.csv', sep=';', index_col=0)
df_solar.index = pd.to_datetime(df_solar.index)

# Chosen country of interest
country='DNK'


labels = ['Onshore wind', 'Offshore wind', 'Solar PV', 'Gas (OCGT)', 'Gas (CCGT)']
model_names = ['Onshore wind', 'Offshore wind', 'Solar', 'OCGT', 'CCGT']
colors=['blue', 'dodgerblue', 'orange', 'crimson', 'darkviolet']

###################################################
######    (b)  Multi weather-year analysis  #######
###################################################
def annuity(n,r):
    """ Calculate the annuity factor for an asset with lifetime n years and discount rate  r """
    if r > 0:
        return r/(1. - 1./(1.+r)**n)
    else:
        return 1/n
    
weather_years = [1979, 1985, 1991, 1995, 1999, 2005, 2010, 2013, 2015]

dict_results = {}
for year in weather_years:
    date_range = pd.date_range(f'{year}-01-01 00:00Z', f'{year}-12-31 23:00Z', freq='h')

    network = pypsa.Network()
    network.set_snapshots(date_range.values)
    network.add("Bus", "electricity bus")
    network.add("Load", "load", bus="electricity bus", p_set=df_elec[country].values)
    network.loads_t.p_set
    network.add("Carrier", "gas", co2_emissions=0.19) # in t_CO2/MWh_th
    network.add("Carrier", "onshorewind")
    network.add("Carrier", "offshorewind")
    network.add("Carrier", "solar")
    # add onshore wind generator
    CF_wind = df_onshorewind[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
    capital_cost_onshorewind = annuity(27,0.07)*1118775 # has been updated
    network.add("Generator", "Onshore wind", bus="electricity bus", p_nom_extendable=True, carrier="onshorewind", capital_cost = capital_cost_onshorewind, marginal_cost = 0, p_max_pu = CF_wind.values)
    # add offshore wind generator
    CF_wind_off = df_offshorewind[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
    capital_cost_offshorewind = annuity(27,0.07)*2115944 # has been updated
    network.add("Generator", "Offshore wind", bus="electricity bus", p_nom_extendable=True, carrier="offshorewind", capital_cost = capital_cost_offshorewind, marginal_cost = 0, p_max_pu = CF_wind_off.values)
    # add solar PV generator
    CF_solar = df_solar[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
    capital_cost_solar = annuity(25,0.07)*450000 # has been updated
    network.add("Generator", "Solar", bus="electricity bus",p_nom_extendable=True, carrier="solar", capital_cost = capital_cost_solar, marginal_cost = 0, p_max_pu = CF_solar.values)
    # add OCGT (Open Cycle Gas Turbine) generator
    capital_cost_OCGT = annuity(25,0.07)*453960  # has been updated
    fuel_cost = 30 # €/MWh
    efficiency_OCGT = 0.41 # MWh_e/MWh
    marginal_cost_OCGT = fuel_cost/efficiency_OCGT # in €/MWh_e
    network.add("Generator", "OCGT", bus="electricity bus", p_nom_extendable=True, carrier="gas", capital_cost = capital_cost_OCGT, marginal_cost = marginal_cost_OCGT)
    # add CCGT
    capital_cost_CCGT = annuity(25,0.07)*880000  # has been updated
    fuel_cost = 30 # €/MWh
    efficiency_CCGT = 0.56 # MWh_e/MWh
    marginal_cost_CCGT = fuel_cost/efficiency_CCGT # in €/MWh_el
    network.add("Generator", "CCGT", bus="electricity bus", p_nom_extendable=True, carrier="gas", capital_cost = capital_cost_CCGT, marginal_cost = marginal_cost_CCGT)
    # Solve the model (optimize the system)
    network.optimize(solver_name='gurobi',solver_options={"OutputFlag": 0,"LogToConsole": 0})


    # Saving installed capacities
    print(year)
    dict_results[year] = network.generators.p_nom_opt
    
# Export dict_results to csv file
df = pd.DataFrame(dict_results).T
df.to_csv('installed_capacities_weather_years.csv')

from matplotlib.lines import Line2D
x = np.arange(len(model_names))
X = np.tile(x, (len(df), 1)).ravel()
Y = df.to_numpy().ravel()
m, s = df.mean(), df.std()

fig, ax = plt.subplots(figsize=(8,4), dpi=300)
ax.scatter(X, Y, s=35, alpha=0.7, label="Weather years")
ax.scatter(x, m.to_numpy(), marker="*", s=220, edgecolor="k", linewidth=0.8, label="Mean")
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylabel("Installed capacity (MW)")
ax.grid(axis="y", alpha=0.3)
leg1 = ax.legend(loc="lower right", frameon=True)
ax.add_artist(leg1)
handles = [Line2D([], [], linestyle="none", label=f"{tech}: {m[tech]:.0f}±{s[tech]:.0f} MW")  for tech in model_names]
leg = ax.legend(handles=handles, loc="upper right", frameon=True, title="Mean ± std")
leg.get_title().set_fontweight("bold")
plt.title("Installed capacities across different weather years", fontweight="bold")
plt.tight_layout()
plt.savefig('1b_installed_capacities_weather_years.png', dpi=300)
plt.show()



