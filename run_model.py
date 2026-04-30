import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Setting_Parameters import Param_Init
from generate_daily_contacts import generate_daily_contacts
from load_profile import build_daily_load, baseline_peak_tuples
from agent import Agent, appliance_shift_rates, default_epsilon_habit, default_epsilon_price, default_epsilon_social
from price_estimator import hour_price_estimator, price_baseline
from metrics import find_local_price_minima, compute_social_targets_for_agent, compile_agent_day_metrics, compile_day_metrics, build_dataframes

"""
This file will be the main simulation entry point for the ABM
It combines all other files:
    -> Setting_Parameters.py: behavioral parameter sampling from group specific beta distribution
    -> load_profile.py: appliance data, distributions, daily load generation
    -> agent.py: Agent class, habit and use characteristics initialization, daily shifting
    -> generate_daily_contacts.py: daily sub-network sampling from the full pre-built network
    -> price_estimator.py: dynamic pricing based on demand/elasticity from green energy
    -> metrics.py: all metric computation

Simulation flow per day:
    Day 0:
        -> Agents already have habit-adjusted peaks (heights and widths only, centers unchanged)
        -> EPEX baseline prices are used because there is nothing to shift upon
        -> Daily contact network is generated and stored for use on day 1
        -> Loads are simulated and day-0 metrics are collected
        -> Prices for day 1 are computed from day 0 aggregate demand

    Until day n:
        -> Each agent's current_peak_lists is saved to previous_peak_lists 
        -> Price and social shifts are computed and applied using those lists
        -> Loads are simulated with the newly shifted distributions
        -> Prices for day X are computed from dayX-1 aggregate demand
"""


def validate_network_code(code):
    """
    Validate a network code string and return the n
    Raises a ValueError with if the code does not match the format
    Valid examples: 50a, 500c, 1000e
    Invalid examples: "200t" (t not in a-e), "201a" (201 not a multiple of 50)
    """
    valid_sizes = list(range(50, 1001, 50))
    valid_variants = ["a", "b", "c", "d", "e"]

    all_combinations = []
    for size in valid_sizes:
        for variant in valid_variants:
            all_combinations.append(f"{size}{variant}")

    if code not in all_combinations:
        raise ValueError(
            f"{code} is not a valid network code. "
            f"Must be a multiple of 50 (50–1000) followed by a variant (a–e). "
            f"Examples: '50a', '500c', '1000e'.")

    n = int(code[:-1]) #get indices starting before last one
    return n 


def load_network(network_code, networks_path="networks.json"):
    """
    Load the full social network for the given code

    The networks.json file is pre-built offline using make_network.py in the groundwork folder

    Parameters:
    -> network_code: e.g. "500a"
    -> networks_path: path to the networks file

    Returns a tuple with the full network and the len of that network
    """
    n = validate_network_code(network_code)  #validate and extract agent count/file

    with open(networks_path, "r") as f:
        all_networks = json.load(f)  #load all networks from the json
    full_network = all_networks[network_code]  
    return full_network, n


#-------------------------
#Main simulation function
#-------------------------

def run_model(
    agents_pct = [80, 10, 10],
    network_code = "500a",
    days = 30,
    graphs = None,
    median_plot = True,
    random_state = 42,
    epsilon_habit = default_epsilon_habit,
    epsilon_price = default_epsilon_price,
    epsilon_social = default_epsilon_social,
    networks_path = "networks.json"):
    """
    Run the full model with behavioral shifting if any epsilon > 0

    Parameters:
    -> agents_pct: list of 3 ints that must sum to 100
        -> Percentage split [Habit-driven%, Price-responsive%, Social-influenced%] (position matters!)
        -> The actual agent count N is determined by the network_code
        -> [80, 10, 10] means 80% habit-driven, 10% price-responsive, 10% social

    -> network_code: key into networks.json, e.g. 500a
    
    -> days: number of days to simulate 

    -> graphs: list of integers or None
        -> [1, 7, 14] prints aggregate plots for days 1, 7, and 14, will skip if exceeds days
        
    -> median_plot: If true, print a median aggregate load profile across all simulated days
       -> also overlays the day 1 and last day aggregate curves for comparison

    -> random_state: controlling seed, same value always produces identical simulation output

    -> epsilon_habit: height bonus per unit of habit_str applied once at initialization
        -> Default is set in agent.py as default_epsilon_habit
        -> Tune to control how sharply peaked agents' preferred usage times are initialized

    -> epsilon_price: price shift scaling factor applied daily
        -> Default is set in agent.py as default_epsilon_price
        -> Higher values make agents shift peaks more strongly toward cheap hours

    -> epsilon_social: social shift scaling factor applied daily
        -> Default is set in agent.py as default_epsilon_social
        -> Higher values make agents peak times draw nearer to social network

    -> networks_path: path to networks.json as default

    Returns:
    -> df_agent_daily: pd.DataFrame -> one row per agent per day, used for RQ1 group analysis
    -> df_daily: pd.DataFrame -> one row per day, used for RQ2 system analysis
    -> load_profiles: np.ndarray -> shape (days, 96), aggregate kW per 15-min slot
    -> df_pricing: pd.DataFrame -> one row per day per hour, columns: day, hour, price_baseline, price_used, price_delta

    """

    agents_pct = list(agents_pct)  #to ensure it is a mutable list
    if len(agents_pct) != 3:
        raise ValueError("agents_pct must have exactly 3 elements: [habit, price, social]")
    if sum(agents_pct) != 100:
        raise ValueError(f"agents_pct must sum to 100. Got {agents_pct} (sum={sum(agents_pct)})")
    if days < 1:
        raise ValueError("days must be at least 1!")

    print(f"Loading network {network_code} from {networks_path}...")
    full_network, n = load_network(network_code, networks_path) #load the json
    agent_ids = list(full_network.keys())   #ordered list of all agent IDs from the JSON
    print(f"->  {n} agents loaded.")
    
    #assigning percentages to n
    habit_count = round(n * agents_pct[0] / 100)          
    price_count = round(n * agents_pct[1] / 100)          
    social_count = n - habit_count - price_count  #social group will take remainder

    print(f"  Group split: {habit_count} Habit-driven, {price_count} Price-responsive, {social_count} Social-influenced  (total: {habit_count + price_count + social_count})")


    master_rng = np.random.default_rng(random_state) #master generator for deriving all sub seeds
    agent_seeds = master_rng.integers(0, 10000000, size=n) #one seed per agent
    day_seeds = master_rng.integers(0, 10000000, size=days) #one seed per simulated day (to set daily contacts)

    params_df = Param_Init(habit_count, price_count, social_count, random_state=random_state)

    print("Initialising agents...")
    agents = []  #list of agent objects
    agents_by_id = {} #dict for quick lookup of agent objects by ID

    for i, agent_id in enumerate(agent_ids):
        row = params_df.iloc[i]                           #get behavioral parameters for this agent
        agent_rng = np.random.default_rng(int(agent_seeds[i])) #agent-specific RNG from its derived seed
        agent_obj = Agent(
            agent_id = agent_id,
            dominant_group = row["dominant_group"],
            habit_str = row["habit_str"],
            price_sens = row["price_sens"],
            soc_suc = row["soc_suc"],
            rng = agent_rng,
            epsilon_habit = epsilon_habit)
        agents.append(agent_obj)
        agents_by_id[agent_id] = agent_obj  #also store in lookup dict

    appliance_names = list(baseline_peak_tuples.keys())

    #Simulation storage
    all_aggregates = []  #list of all 96 slots, one aggregate load profile per day
    all_daily_profiles = []  #list of lists, all agent load arrays per day
    agent_day_records = []  #will become df_agent_daily in metrics.py
    day_records = []  #will become df_daily in metrics.py
    pricing_records = []  #will become df_pricing, one row per day per hour

    #day 0 uses EPEX baseline prices
    current_prices_24h = list(price_baseline)

    #contact network from the previous day, used for social shift target computation starts as None (on day 0)
    previous_day_contacts = None

    #Main simulation loop
    print(f"\nRunning simulation: {days} days, {n} agents, seed {random_state}\n"
          f"  epsilon_habit={epsilon_habit}  epsilon_price={epsilon_price}  "
          f"epsilon_social={epsilon_social}\n")

    for day in range(days):
        is_last_day = (day == days - 1)

        #collect prices used today for df_pricing, before they get updated at the end of this iteration
        for hour in range(24):
            pricing_records.append({
                "day": day,
                "hour": hour,
                "price_baseline": price_baseline[hour],
                "price_used": current_prices_24h[hour],
                "price_delta": round(current_prices_24h[hour] - price_baseline[hour], 3)})

        #generate daily contact sub-network        
        today_contacts = generate_daily_contacts(
            full_network = full_network,
            day_seed = int(day_seeds[day]))
        
        #apply price and social shifts (skipped on day 0)
        if day > 0:
            #save current appliance peaks as for all agents
            for agent in agents:
            #initialize an empty dictionary for the previous peaks
                agent.previous_peak_lists = {}
                #iterate through each key and value in the current peak lists
                for k, v in agent.current_peak_lists.items():
                    agent.previous_peak_lists[k] = list(v)
                
            #find local price minima in today's price curve
            price_minima = find_local_price_minima(current_prices_24h)

            #compute social targets for every agent
            #Each agent looks at the previous_peak_lists of its DAY-1 contacts
            social_targets_all = {}
            for agent in agents:
                social_targets_all[agent.agent_id] = compute_social_targets_for_agent(
                    agent = agent,
                    previous_day_contacts = previous_day_contacts,
                    agents_by_id = agents_by_id,
                    appliance_names = appliance_names)

            #apply price and social shifts to all agents
            for agent in agents:
                agent.apply_shifts(
                    price_minima = price_minima,
                    social_targets = social_targets_all[agent.agent_id],
                    epsilon_price = epsilon_price,
                    epsilon_social = epsilon_social)

        
        #Simulate one day of load for every agent
        day_profiles = []  #list of agents' load array tuples for this day
        today_agent_records = []  #agent metric dicts for this day only

        for agent in agents:
            load, overflow, _ = build_daily_load(
                agent_appliances = agent.appliance_chars,
                has_ev = agent.has_ev,
                random_state = agent.rng,
                previous_overflow = agent.previous_overflow,
                custom_distributions = agent.current_distributions)
            agent.previous_overflow = overflow  #carry overflow forward to tomorrow
            day_profiles.append((agent, load))  #store the agent and its load together for later

        aggregate = np.zeros(96)           #init as zero array
        for _, load in day_profiles:
            aggregate += load              #add each agents own load to the total

        all_aggregates.append(aggregate)   #save aggregate for plots and output
        
        current_day_loads = []
        #Iterate through the pairs in day_profiles
        for _, load in day_profiles:
            current_day_loads.append(load)
        all_daily_profiles.append(current_day_loads)
        
        #Compute prices for the next day from today's aggregate demand
        #price estimator works with hourly demand/n 
        #average the 4 quarters per hour then divide by n to get kW per agent per hour

        hourly_per_agent = aggregate.reshape(24, 4).mean(axis=1) / n  #convert to hourly per-agent demand
        next_prices_24h = hour_price_estimator(hourly_per_agent)      #estimate tomorrow's prices

        #Now collect metrics for this day
        prices = np.zeros(96)
        for s in range(96):
            #calculate hour index
            hour_index = s // 4
            #extract the price for that hour and assign it to the slot
            prices[s] = current_prices_24h[hour_index]
        for agent, load in day_profiles:
            record = compile_agent_day_metrics(
                agent = agent,
                day = day,
                load = load,
                prices = prices,
                is_last_day = is_last_day) 
            today_agent_records.append(record)  #needed for compile_day_metrics below
            agent_day_records.append(record) #appended to the full list for df_agent_daily

        day_record = compile_day_metrics(
            day = day,
            aggregate = aggregate,
            prices = current_prices_24h,
            agent_records = today_agent_records)
        day_records.append(day_record) #appended to the full list for df_daily

        #print a summary line for each day so progress is visible
        print(
            f"Day {day + 1} / {days} | "
            f"peak: {aggregate.max():.2f} kW | "
            f"mean: {aggregate.mean():.2f} kW | "
            f"PAR: {day_record['par']:.2f} | "
            f"flex: {day_record['total_flexibility']:.2f} | "
            f"norm_flex: {day_record['total_flexibility']/n:.2f} | " 
            f"price_mean: {np.mean(current_prices_24h):.2f}")

        current_prices_24h = next_prices_24h   #tomorrow uses today's estimated prices
        previous_day_contacts = today_contacts  #tomorrow's social shift uses today's contact network

    print("\nBuilding output DataFrames...")
    df_agent_daily, df_daily = build_dataframes(agent_day_records, day_records)

    df_pricing = pd.DataFrame(pricing_records)
    load_profiles = np.array(all_aggregates) 

    #Plotting    
    time_axis = np.linspace(0, 24, 96, endpoint=False)  #24-hour x-axis with 96 points

    #individual day plots requested by the generator (graphs parameter is 1-indexed)
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
                f"Aggregate load - Day {day_number}  "
                f"({agents_pct} composition, network - {network_code}, seed {random_state})")
            ax.grid(alpha=0.3)
            plt.xticks(range(25))
            plt.tight_layout()
            plt.show()

    #median profile across the entire simulation period, overlaid with day 1 and last day
    if median_plot:
        median_profile = np.median(load_profiles, axis=0)  #element-wise median across all days
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(time_axis, median_profile, color="lightblue", alpha=0.7, label="Median")
        ax.plot(time_axis, all_aggregates[0], color="darkblue", linewidth=1.2, alpha=0.7, label="Day 1")
        ax.plot(time_axis, all_aggregates[-1], color="darkred", linewidth=1.2, alpha=0.7, label=f"Day {days}")
        ax.set_ylabel("kW")
        ax.set_xlabel("Hour of day")
        ax.set_title(
            f"Median aggregate load profile - {days} days, "
            f"{agents_pct} composition, network '{network_code}', seed {random_state}"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(range(25))
        plt.tight_layout()
        plt.show()

    print("\nSimulation complete.")
    print(f"  df_agent_daily : {df_agent_daily.shape}  (agents x days = {n} x {days})")
    print(f"  df_daily       : {df_daily.shape}        (one row per day)")
    print(f"  load_profiles  : {load_profiles.shape}   (days x 96 slots)")
    print(f"  df_pricing     : {df_pricing.shape}  ({days} days x 24 hours)")   

    return df_agent_daily, df_daily, load_profiles, df_pricing
