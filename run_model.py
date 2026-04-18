import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Setting_Parameters      import Param_Init
from generate_daily_contacts import generate_daily_contacts
from load_profile            import build_daily_load, baseline_peak_tuples
from agent                   import Agent, appliance_shift_rates, default_epsilon_habit, default_epsilon_price, default_epsilon_social
from price_estimator         import hour_price_estimator, price_baseline
from metrics                 import (
    find_local_price_minima,
    compute_social_targets_for_agent,
    compile_agent_day_metrics,
    compile_day_metrics,
    build_dataframes,
)

"""
run_model.py — Main simulation entry point for the residential electricity ABM.

This module ties all components together:
    -> Setting_Parameters.py     : behavioral parameter sampling from Beta distributions
    -> load_profile.py           : appliance data, distributions, and daily load generation
    -> agent.py                  : Agent class, habit initialization, and daily shifting
    -> generate_daily_contacts.py: daily sub-network sampling from the full pre-built network
    -> price_estimator.py        : EPEX-based dynamic pricing from yesterday's demand
    -> metrics.py                : all metric computation and DataFrame construction

Usage:
    from run_model import run_model

    df_agents, df_daily, load_profiles = run_model(
        agents_pct   = [80, 10, 10],
        network_code = "500a",
        days         = 30,
        graphs       = [1, 7, 14, 30],
        median_plot  = True,
        random_state = 42,
    )

Network JSON format:
    networks.json must sit in the same directory as run_model.py
    Top-level keys are network codes (e.g. "500a")
    Each value is a dict: {agent_id: [neighbor_id, ...]}
    Agent IDs must be strings (e.g. "AG001")

Valid network codes:
    Numeric part: multiple of 50, range [50, 1000]
    Letter part : one of {a, b, c, d, e}
    Examples    : "50a", "500c", "1000e"

Simulation flow per day:
    Day 0:
        -> Agents already have habit-adjusted peaks from Agent.__init__
        -> EPEX baseline prices are used (no demand-response on day 0)
        -> Day-0 daily contact network is generated and stored for use on day 1
        -> Loads are simulated and day-0 metrics are collected
        -> Prices for day 1 are computed from day-0 aggregate demand

    Day d > 0:
        -> Each agent's current_peak_lists is saved to previous_peak_lists (snapshot)
        -> Price and social shifts are computed and applied using those snapshots
        -> Loads are simulated with the newly shifted distributions
        -> Prices for day d+1 are computed from day-d aggregate demand
"""


#-----------------------------------------------------------------------
#Network loading and validation
#-----------------------------------------------------------------------

valid_sizes    = list(range(50, 1001, 50))  #[50, 100, 150, ..., 1000]
valid_variants = list("abcde")              #a, b, c, d, e


def _validate_network_code(code):
    """
    Validate a network code string and return (N, variant).
    Raises a ValueError with a clear message if the code does not match the expected format.

    Valid examples: "50a", "500c", "1000e"
    Invalid examples: "200t" (t not in a-e), "201a" (201 not a multiple of 50)
    """
    match = re.fullmatch(r"(\d+)([a-e])", str(code))  #must be digits followed by a single letter a-e
    if not match:
        raise ValueError(
            f"Invalid network code '{code}'. "
            f"Expected a number followed by a letter a-e, e.g. '500a' or '150c'."
        )
    size    = int(match.group(1))   #extract the numeric part
    variant = match.group(2)        #extract the letter part
    if size not in valid_sizes:
        raise ValueError(
            f"Network size {size} is not valid. "
            f"Must be a multiple of 50 between 50 and 1000."
        )
    return size, variant


def _load_network(network_code, networks_path="networks.json"):
    """
    Load the full adjacency network for the given network_code from networks.json.

    The networks.json file is pre-built offline using make_network.py
    and is NOT generated at runtime.

    Parameters:
    - network_code  : str -> e.g. "500a"
    - networks_path : str -> path to the networks JSON file

    Returns a tuple:
    -> dict {agent_id: [neighbor_id, ...]} representing the full neighborhood for each agent
    -> int N -> number of agents in this network
    """
    n, _ = _validate_network_code(network_code)  #validate and extract the implied agent count

    with open(networks_path, "r") as f:
        all_networks = json.load(f)  #load all networks from the JSON file

    if network_code not in all_networks:
        available = list(all_networks.keys())
        raise KeyError(
            f"Network code '{network_code}' not found in {networks_path}. "
            f"Available codes: {available}"
        )

    full_network = all_networks[network_code]  #extract the adjacency dict for this code
    actual_n     = len(full_network)           #count how many agents are actually in the JSON

    if actual_n != n:
        #warn if the JSON count differs from what the code implies, then use the actual count
        print(
            f"Warning: network code '{network_code}' implies {n} agents "
            f"but the JSON contains {actual_n}. Using actual count ({actual_n})."
        )

    return full_network, actual_n


#-----------------------------------------------------------------------
#Main simulation function
#-----------------------------------------------------------------------

def run_model(
    agents_pct         = (80, 10, 10),
    network_code       = "500a",
    days               = 30,
    graphs             = None,
    median_plot        = True,
    random_state       = 42,
    epsilon_habit      = default_epsilon_habit,
    epsilon_price      = default_epsilon_price,
    epsilon_social     = default_epsilon_social,
    rebound_prominence = 0.5,
    networks_path      = "networks.json",
):
    """
    Run the full residential electricity ABM simulation with behavioral shifting.

    Parameters:
    - agents_pct    : tuple or list of 3 ints that must sum to 100
        -> Percentage split [Habit-driven%, Price-responsive%, Social-influenced%]
        -> The actual agent count N is determined by the network_code, not by this parameter
        -> Example: [80, 10, 10] means 80% habit-driven, 10% price-responsive, 10% social

    - network_code  : str -> key into networks.json, e.g. "500a"
        -> Determines N (total agents) and the full social network topology
        -> Valid format: multiple of 50 (range 50-1000) followed by a letter a-e

    - days          : int -> number of days to simulate (day 0 is the first, no shifting applied)

    - graphs        : list of int or None
        -> 1-indexed day numbers for which to print an aggregate load plot
        -> Example: [1, 7, 14] prints plots for days 1, 7, and 14
        -> Pass None to skip all individual day plots

    - median_plot   : bool -> if True, print a median aggregate load profile across all simulated days

    - random_state  : int -> master seed, same value always produces identical simulation output

    - epsilon_habit  : float -> height bonus per unit of habit_str applied once at initialization
        -> Default is set in agent.py as default_epsilon_habit
        -> Tune here to control how sharply peaked agents' preferred usage times are

    - epsilon_price  : float -> price shift scaling factor applied daily
        -> Default is set in agent.py as default_epsilon_price
        -> Higher values make agents shift peaks more strongly toward cheap hours

    - epsilon_social : float -> social shift scaling factor applied daily
        -> Default is set in agent.py as default_epsilon_social
        -> Higher values make agents converge more quickly toward their neighbors' schedules

    - rebound_prominence : float -> prominence threshold for counting rebound peaks
        -> A peak must exceed this fraction of the day's mean load above its surroundings
        -> Default 0.5 means a peak must stand at least 50% of mean load above neighbors

    - networks_path : str -> path to networks.json, default assumes it is in the same directory

    Returns:
    - df_agent_daily : pd.DataFrame -> one row per (agent, day), use for RQ1 group analysis
    - df_daily       : pd.DataFrame -> one row per day, use for RQ2 system analysis
    - load_profiles  : np.ndarray of shape (days, 96) -> aggregate kW per 15-min slot per day
    """

    #------------------------------------------------------------------
    #Input validation
    #------------------------------------------------------------------

    agents_pct = list(agents_pct)  #ensure it is a mutable list
    if len(agents_pct) != 3:
        raise ValueError("agents_pct must have exactly 3 elements: [habit%, price%, social%].")
    if sum(agents_pct) != 100:
        raise ValueError(f"agents_pct must sum to 100. Got {agents_pct} (sum={sum(agents_pct)}).")
    if days < 1:
        raise ValueError("days must be at least 1.")

    #------------------------------------------------------------------
    #Network loading
    #------------------------------------------------------------------

    print(f"Loading network '{network_code}' from {networks_path}...")
    full_network, n = _load_network(network_code, networks_path)  #load the pre-built adjacency dict
    agent_ids        = list(full_network.keys())                  #ordered list of all agent IDs from the JSON
    print(f"  {n} agents loaded.")

    #------------------------------------------------------------------
    #Compute group sizes from percentages
    #-> N may not be perfectly divisible by 100, so the remainder is assigned to the social group
    #------------------------------------------------------------------

    habit_count  = round(n * agents_pct[0] / 100)          #number of habit-driven agents
    price_count  = round(n * agents_pct[1] / 100)          #number of price-responsive agents
    social_count = n - habit_count - price_count            #social group absorbs any rounding remainder

    print(f"  Group split: {habit_count} Habit-driven, {price_count} Price-responsive, "
          f"{social_count} Social-influenced  (total: {habit_count + price_count + social_count})")

    #------------------------------------------------------------------
    #Reproducible random state setup
    #-> All seeds are derived from the master random_state
    #-> Passing the same integer always produces identical output
    #------------------------------------------------------------------

    master_rng  = np.random.default_rng(random_state)              #master generator for deriving all sub-seeds
    agent_seeds = master_rng.integers(0, 10_000_000, size=n)       #one seed per agent
    day_seeds   = master_rng.integers(0, 10_000_000, size=days)    #one seed per simulated day (for daily contacts)

    #------------------------------------------------------------------
    #Behavioral parameter sampling
    #-> Param_Init uses its own RNG seeded with random_state
    #-> The same random_state always gives the same parameters regardless of n or days
    #------------------------------------------------------------------

    params_df = Param_Init(habit_count, price_count, social_count, random_state=random_state)

    #------------------------------------------------------------------
    #Agent construction
    #-> Agent IDs come from the network JSON
    #-> Behavioral parameters come from params_df rows in the same order
    #   (habit agents first, then price, then social, matching Param_Init's output order)
    #-> Agent.__init__ calls sample_agent_appliances() and _initialize_peaks()
    #   so the one-time habit shift is applied before the simulation starts
    #------------------------------------------------------------------

    print("Initialising agents...")
    agents       = []          #ordered list of Agent objects
    agents_by_id = {}          #dict for quick lookup of Agent objects by ID

    for i, agent_id in enumerate(agent_ids):
        row       = params_df.iloc[i]                           #get behavioral parameters for this agent
        agent_rng = np.random.default_rng(int(agent_seeds[i])) #agent-specific RNG from its derived seed
        agent_obj = Agent(
            agent_id       = agent_id,
            dominant_group = row["dominant_group"],
            habit_str      = row["habit_str"],
            price_sens     = row["price_sens"],
            soc_suc        = row["soc_suc"],
            rng            = agent_rng,
            epsilon_habit  = epsilon_habit,               #passed in so habit intensity is tunable globally
        )
        agents.append(agent_obj)
        agents_by_id[agent_id] = agent_obj  #also store in lookup dict

    appliance_names = list(baseline_peak_tuples.keys())  #ordered list of appliances, used throughout

    #------------------------------------------------------------------
    #Simulation storage
    #------------------------------------------------------------------

    all_aggregates     = []  #list of np.array(96), one aggregate load profile per day
    all_daily_profiles = []  #list of lists, all agent load arrays per day (for potential post-analysis)
    agent_day_records  = []  #flat list of metric dicts -> will become df_agent_daily
    day_records        = []  #list of daily metric dicts -> will become df_daily

    #day 0 uses EPEX baseline prices, no demand-response yet on the first day
    current_prices_24h = list(price_baseline)

    #contact network from the previous day, used for social shift target computation
    #starts as None because there is no "yesterday" on day 0
    previous_day_contacts = None

    #------------------------------------------------------------------
    #Main simulation loop
    #------------------------------------------------------------------

    print(f"\nRunning simulation: {days} days, {n} agents, seed {random_state}\n"
          f"  epsilon_habit={epsilon_habit}  epsilon_price={epsilon_price}  "
          f"epsilon_social={epsilon_social}\n")

    for day in range(days):

        #--------------------------------------------------------------
        #Step 1: Generate today's daily contact sub-network
        #-> Uses a day-specific seed derived from the master rng
        #-> Contacts differ each day but are always identical for the same random_state
        #--------------------------------------------------------------

        today_contacts = generate_daily_contacts(
            full_network = full_network,
            day_seed     = int(day_seeds[day]),  #reproducible seed for today's contact sampling
        )

        #--------------------------------------------------------------
        #Step 2: Apply price and social shifts (skipped on day 0)
        #-> On day 0 agents already have their habit-adjusted peaks from Agent.__init__
        #-> No shifting occurs until day 1
        #--------------------------------------------------------------

        if day > 0:

            #2a: Save current peaks as "previous" for ALL agents BEFORE any agent is shifted
            #    This snapshot guarantees that social targets are computed from a consistent
            #    previous-day state. No agent reads another agent's already-shifted peaks.
            for agent in agents:
                agent.previous_peak_lists = {
                    k: list(v) for k, v in agent.current_peak_lists.items()
                }

            #2b: Find local price minima in today's price curve
            #    Agents will shift their peaks toward the nearest valley
            price_minima = find_local_price_minima(current_prices_24h)

            #2c: Compute social targets for every agent in a single pass
            #    Each agent looks at the previous_peak_lists of its day-(d-1) contacts
            #    previous_day_contacts was the contact network generated on the previous day
            social_targets_all = {}
            for agent in agents:
                social_targets_all[agent.agent_id] = compute_social_targets_for_agent(
                    agent                 = agent,
                    previous_day_contacts = previous_day_contacts,
                    agents_by_id          = agents_by_id,
                    appliance_names       = appliance_names,
                )

            #2d: Apply price and social shifts to all agents
            #    apply_shifts() updates current_peak_lists, current_distributions,
            #    and the last_*_flexibility trackers in place on each agent
            for agent in agents:
                agent.apply_shifts(
                    price_minima   = price_minima,
                    social_targets = social_targets_all[agent.agent_id],
                    epsilon_price  = epsilon_price,
                    epsilon_social = epsilon_social,
                )

        #--------------------------------------------------------------
        #Step 3: Simulate one day of load for every agent
        #-> Each agent's shifted distributions are passed as custom_distributions
        #   to build_daily_load(), replacing the fixed module-level baselines
        #-> The agent's personal RNG is advanced by each call, maintaining
        #   the same sequence as in the original run_simulation.py
        #--------------------------------------------------------------

        day_profiles        = []  #list of (agent, load_array) tuples for this day
        today_agent_records = []  #agent metric dicts for this day only (used in compile_day_metrics)

        for agent in agents:
            load, overflow, _ = build_daily_load(
                agent_appliances     = agent.appliance_chars,
                has_ev               = agent.has_ev,
                random_state         = agent.rng,
                previous_overflow    = agent.previous_overflow,
                custom_distributions = agent.current_distributions,  #use the agent's shifted distributions
            )
            agent.previous_overflow = overflow  #carry overflow forward to tomorrow
            day_profiles.append((agent, load))  #store the agent and its load together for later

        #--------------------------------------------------------------
        #Step 4: Aggregate load across all agents
        #--------------------------------------------------------------

        aggregate = np.zeros(96)           #start from a zero array
        for _, load in day_profiles:
            aggregate += load              #add each agent's load to the running total

        all_aggregates.append(aggregate)   #save this day's aggregate for plots and output
        all_daily_profiles.append([load for _, load in day_profiles])  #save per-agent profiles

        #--------------------------------------------------------------
        #Step 5: Compute prices for the NEXT day from today's aggregate demand
        #-> The price estimator expects per-agent hourly kW demand
        #-> Reshape the 96 slots to (24, 4), average the 4 quarters per hour,
        #   then divide by n to get kW per agent per hour
        #--------------------------------------------------------------

        hourly_per_agent = aggregate.reshape(24, 4).mean(axis=1) / n  #convert to hourly per-agent demand
        next_prices_24h  = hour_price_estimator(hourly_per_agent)      #estimate tomorrow's prices

        #--------------------------------------------------------------
        #Step 6: Collect metrics for this day
        #-> Build a 96-slot price array once per day for cost computation
        #   (minor speed-up: avoids a prices_24h[s//4] lookup inside the per-agent inner loop)
        #--------------------------------------------------------------

        prices_96 = np.array([current_prices_24h[s // 4] for s in range(96)])  #expand hourly prices to 15-min slots

        for agent, load in day_profiles:
            record = compile_agent_day_metrics(
                agent     = agent,
                day       = day,
                load      = load,
                prices_96 = prices_96,
            )
            today_agent_records.append(record)  #needed for compile_day_metrics below
            agent_day_records.append(record)    #appended to the full list for df_agent_daily

        day_record = compile_day_metrics(
            day           = day,
            aggregate_96  = aggregate,
            prices_24h    = current_prices_24h,
            agent_records = today_agent_records,
            prominence    = rebound_prominence,
        )
        day_records.append(day_record)  #appended to the full list for df_daily

        #print a summary line for each day so progress is visible
        print(
            f"Day {day + 1:>3}/{days}  |  "
            f"peak: {aggregate.max():7.2f} kW  |  "
            f"mean: {aggregate.mean():6.2f} kW  |  "
            f"PAR: {day_record['par']:.3f}  |  "
            f"flex: {day_record['accumulative_flexibility']:.2f}  |  "
            f"price_mean: {np.mean(current_prices_24h):.3f}"
        )

        #--------------------------------------------------------------
        #Step 7: Advance state variables for the next day
        #--------------------------------------------------------------

        current_prices_24h    = next_prices_24h   #tomorrow uses today's estimated prices
        previous_day_contacts = today_contacts     #tomorrow's social shift uses today's contact network

    #------------------------------------------------------------------
    #Build output DataFrames
    #------------------------------------------------------------------

    print("\nBuilding output DataFrames...")
    df_agent_daily, df_daily = build_dataframes(agent_day_records, day_records)
    load_profiles             = np.array(all_aggregates)  #shape: (days, 96)

    #------------------------------------------------------------------
    #Plotting
    #------------------------------------------------------------------

    time_axis = np.linspace(0, 24, 96, endpoint=False)  #24-hour x-axis with 96 points

    #individual day plots requested by the user (graphs parameter is 1-indexed)
    if graphs is not None:
        for day_number in graphs:
            day_index = day_number - 1  #convert to 0-indexed to match all_aggregates
            if day_index < 0 or day_index >= days:
                print(f"Warning: graph day {day_number} out of range [1, {days}], skipped.")
                continue
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.fill_between(time_axis, all_aggregates[day_index], color="salmon", alpha=0.7)
            ax.set_ylabel("kW")
            ax.set_xlabel("Hour of day")
            ax.set_title(
                f"Aggregate load — Day {day_number}  "
                f"({n} agents, network '{network_code}', seed {random_state})"
            )
            ax.grid(alpha=0.3)
            plt.xticks(range(25))
            plt.tight_layout()
            plt.show()

    #median profile across the entire simulation period
    if median_plot:
        median_profile = np.median(load_profiles, axis=0)  #element-wise median across all days
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(time_axis, median_profile, color="lightblue", alpha=0.7)
        ax.set_ylabel("kW")
        ax.set_xlabel("Hour of day")
        ax.set_title(
            f"Median aggregate load profile — {days} days, "
            f"{n} agents, network '{network_code}', seed {random_state}"
        )
        ax.grid(alpha=0.3)
        plt.xticks(range(25))
        plt.tight_layout()
        plt.show()

    print("\nSimulation complete.")
    print(f"  df_agent_daily : {df_agent_daily.shape}  (agents x days = {n} x {days})")
    print(f"  df_daily       : {df_daily.shape}        (one row per day)")
    print(f"  load_profiles  : {load_profiles.shape}   (days x 96 slots)")

    return df_agent_daily, df_daily, load_profiles