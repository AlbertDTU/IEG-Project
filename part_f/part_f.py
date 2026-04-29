import os
import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt

# ── Load data ──────────────────────────────────────────────────────────────────
df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0)
df_elec.index = pd.to_datetime(df_elec.index)

df_onshorewind = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0)
df_onshorewind.index = pd.to_datetime(df_onshorewind.index)

df_offshorewind = pd.read_csv('data/offshore_wind_1979-2017.csv', sep=';', index_col=0)
df_offshorewind.index = pd.to_datetime(df_offshorewind.index)

df_solar = pd.read_csv('data/pv_optimal.csv', sep=';', index_col=0)
df_solar.index = pd.to_datetime(df_solar.index)

country = 'DNK'

# ── Helper ─────────────────────────────────────────────────────────────────────
def annuity(n, r):
    if r > 0:
        return r / (1. - 1. / (1. + r) ** n)
    else:
        return 1 / n

# ── Build the base network (no CO2 constraint yet) ────────────────────────────
def build_network():
    network = pypsa.Network()
    hours_in_2015 = pd.date_range('2015-01-01 00:00Z', '2015-12-31 23:00Z', freq='h')
    network.set_snapshots(hours_in_2015.values)
    network.add("Bus", "electricity bus")
    network.add("Load", "load", bus="electricity bus", p_set=df_elec[country].values)

    network.add("Carrier", "gas", co2_emissions=0.19)
    network.add("Carrier", "onshorewind")
    network.add("Carrier", "offshorewind")
    network.add("Carrier", "solar")
    network.add("Carrier", "battery storage")

    CF_wind = df_onshorewind[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ") for h in network.snapshots]]
    network.add("Generator", "Onshore wind", bus="electricity bus",
                p_nom_extendable=True, carrier="onshorewind",
                capital_cost=annuity(27, 0.07) * 1118775, marginal_cost=0,
                p_max_pu=CF_wind.values)

    CF_wind_off = df_offshorewind[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ") for h in network.snapshots]]
    network.add("Generator", "Offshore wind", bus="electricity bus",
                p_nom_extendable=True, carrier="offshorewind",
                capital_cost=annuity(27, 0.07) * 2115944, marginal_cost=0,
                p_max_pu=CF_wind_off.values)

    CF_solar = df_solar[country][[h.strftime("%Y-%m-%dT%H:%M:%SZ") for h in network.snapshots]]
    network.add("Generator", "Solar", bus="electricity bus",
                p_nom_extendable=True, carrier="solar",
                capital_cost=annuity(25, 0.07) * 450000, marginal_cost=0,
                p_max_pu=CF_solar.values)

    network.add("Generator", "OCGT", bus="electricity bus",
                p_nom_extendable=True, carrier="gas",
                capital_cost=annuity(25, 0.07) * 453960,
                marginal_cost=30 / 0.41)

    network.add("Generator", "CCGT", bus="electricity bus",
                p_nom_extendable=True, carrier="gas",
                capital_cost=annuity(25, 0.07) * 880000,
                marginal_cost=30 / 0.56)

    network.add("StorageUnit", "battery storage",
                bus="electricity bus", carrier="battery storage",
                max_hours=2,
                capital_cost=annuity(20, 0.07) * 2 * 288000,
                efficiency_store=0.98, efficiency_dispatch=0.97,
                p_nom_extendable=True, cyclic_state_of_charge=True)

    return network

# ── CO2 sweep ─────────────────────────────────────────────────────────────────
# Range: from 9 Mt down to 0.5 Mt
co2_limits = np.linspace(9e6, 0, 15)


gen_cols   = ["Onshore wind", "Offshore wind", "Solar", "OCGT", "CCGT", "battery storage"]
cap_cols   = ["Onshore wind", "Offshore wind", "Solar", "OCGT", "CCGT", "battery storage"]

capacity_results   = {}
generation_results = {}
cost_results       = {}
emission_results   = {}

network = build_network()

for co2_val in co2_limits:
    # Update the CO2 constraint
    if "co2_limit" in network.global_constraints.index:
        network.remove("GlobalConstraint", "co2_limit")
    network.add("GlobalConstraint", "co2_limit",
                type="primary_energy",
                carrier_attribute="co2_emissions",
                sense="<=",
                constant=co2_val)

    try:
        status, condition = network.optimize(solver_name='gurobi', solver_options={"OutputFlag": 0, "LogToConsole": 0})
    except Exception:
        print(f"  Infeasible at {co2_val/1e6:.2f} Mt — skipping")
        continue

    if status != "ok":
        print(f"  Solver status '{status}' at {co2_val/1e6:.2f} Mt — skipping")
        continue

    # Capacity [GW]
    cap = network.generators.p_nom_opt.copy()
    cap["battery storage"] = network.storage_units.p_nom_opt["battery storage"]
    capacity_results[co2_val] = cap[cap_cols].div(1e3)

    # Generation [TWh/yr]
    gen = network.generators_t.p.sum().copy()
    gen["battery storage"] = network.storage_units_t.p.clip(lower=0).sum()["battery storage"]
    generation_results[co2_val] = gen[gen_cols].div(1e6)

    # Total system cost [M€/yr]
    cost_results[co2_val] = network.objective / 1e6

    # CO2 emissions per gas generator [Mt CO2/yr]
    # primary_energy = generation / efficiency * co2_emissions_factor (0.19 t/MWh_th)
    emission_results[co2_val] = pd.Series({
        "OCGT": network.generators_t.p["OCGT"].sum() / 0.41 * 0.19 / 1e6,
        "CCGT": network.generators_t.p["CCGT"].sum() / 0.56 * 0.19 / 1e6,
    })

    print(f"  CO2 = {co2_val/1e6:.2f} Mt  |  cost = {cost_results[co2_val]:.1f} M€")

# ── Assemble DataFrames ────────────────────────────────────────────────────────
df_capacity   = pd.DataFrame(capacity_results).T    # index: tonnes, cols: GW
df_generation = pd.DataFrame(generation_results).T  # index: tonnes, cols: TWh
df_cost       = pd.Series(cost_results)              # index: tonnes, values: M€
df_emissions  = pd.DataFrame(emission_results).T     # index: tonnes, cols: Mt CO2

# Convert index to Mt CO2 for readable axis labels
df_capacity.index   = df_capacity.index / 1e6
df_generation.index = df_generation.index / 1e6
df_cost.index       = df_cost.index / 1e6
df_emissions.index  = df_emissions.index / 1e6

TECH_ORDER  = gen_cols
TECH_LABELS = ['Onshore wind', 'Offshore wind', 'Solar PV', 'Gas (OCGT)', 'Gas (CCGT)', 'Battery storage']
COLORS      = ['blue', 'dodgerblue', 'orange', 'crimson', 'darkviolet', 'lightgreen']

CO2_REF = 8.27   # Mt — Denmark EU ETS 2024 verified emissions
CO2_BIND = 3.3   # Mt — approximate point where CO2 constraint becomes binding

# Sort DataFrames high→low CO2 for the area charts
df_generation = df_generation.sort_index(ascending=False)
df_capacity   = df_capacity.sort_index(ascending=False)


os.makedirs('plots_part_f', exist_ok=True)


def plot_stacked_area(df, ylabel, title, filename):
    _, ax = plt.subplots(figsize=(10, 5), dpi=300)

    x = df.index.values
    arrays = [df[col].values for col in TECH_ORDER]

    ax.stackplot(x, *arrays, labels=TECH_LABELS, colors=COLORS, linewidth=0)
    ax.axvline(x=CO2_REF, color='black', linestyle='--', linewidth=1.2,
               label=f'{CO2_REF} Mt limit')
    ax.axvline(x=CO2_BIND, color='grey', linestyle=':', linewidth=1.5,
               label=f'Constraint binding ({CO2_BIND} Mt)')
    ymax = ax.get_ylim()[1]
    ax.text(CO2_BIND + 0.15, ymax * 0.95, f'{CO2_BIND} Mt\n(binding)',
            fontsize=8, color='grey', fontweight='bold',
            ha='left', va='top')

    ax.set_xlabel('CO\u2082 limit [Mt CO\u2082/yr]', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlim(x[0], x[-1])
    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles, labels_, bbox_to_anchor=(1.01, 1), loc='upper left',
              frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'  Saved {filename}')


plot_stacked_area(
    df_generation,
    ylabel='Annual generation [TWh/yr]',
    title='Generation mix vs. CO\u2082 constraint (Denmark)',
    filename='plots_part_f/part_f_generation_mix.png',
)

plot_stacked_area(
    df_capacity,
    ylabel='Installed capacity [GW]',
    title='Capacity mix vs. CO\u2082 constraint (Denmark)',
    filename='plots_part_f/part_f_capacity_mix.png',
)
