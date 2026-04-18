import numpy as np
import pandas as pd
from scipy.signal import find_peaks

"""
metrics.py — Metric computation for the residential electricity ABM.

This module provides pure functions (no side effects) that compute all tracked metrics
from simulation data. Two DataFrames are assembled and returned to run_model.py:

    df_agent_daily -> one row per (agent, day), enables group-level analysis for RQ1:
        "How do different household behavioral decision architectures affect flexibility
         and cost-flexibility tradeoffs?"

    df_daily -> one row per day, enables system-level analysis for RQ2:
        "How do different configurations of heterogeneous households influence aggregate
         load patterns, peak formation, and overall grid stability?"

All functions are called from run_model.py at the end of each simulated day.

Cost metric design note:
    -> Costs are computed on appliance load ONLY (the always-on baseline is subtracted)
    -> This removes the component that cannot be shifted through behavior
    -> Only behaviorally driven energy expenditure is captured in the cost metric
    -> The normalized cost (pricing_units per kWh of appliance load) is comparable across
       agents regardless of total household size or consumption level
    -> Pricing units are EPEX-derived values divided by 10 as set in price_estimator.py
    -> They are internally consistent for relative comparison but do not map to consumer euros
"""


#-----------------------------------------------------------------------
#Helper: local price minima
#Used by run_model.py to determine which hours agents should shift their peaks toward
#-----------------------------------------------------------------------

def find_local_price_minima(prices_24h):
    """
    Find local minima in a 24-hour price curve.

    Uses scipy.signal.argrelmin with order=1 so a minimum requires the price to be
    strictly lower than both immediate neighbors.
    Falls back to the global minimum if no local minima are detected (e.g. flat curve).

    Returns a list of integer hour indices.
    """
    from scipy.signal import argrelmin
    prices_arr = np.array(prices_24h, dtype=float)  #convert to numpy array for argrelmin
    minima_idx = argrelmin(prices_arr, order=1)[0].tolist()  #find all strict local minima
    if not minima_idx:
        #fallback: treat the single cheapest hour as the only target
        #this can happen with a monotone or extremely flat price curve
        minima_idx = [int(np.argmin(prices_arr))]
    return minima_idx


#-----------------------------------------------------------------------
#Helper: social targets
#Called in run_model.py before apply_shifts() for each agent
#-----------------------------------------------------------------------

def compute_social_targets_for_agent(agent, previous_day_contacts, agents_by_id, appliance_names):
    """
    Compute per-appliance social shift targets for one agent.

    For each appliance, this function collects all peak centers from yesterday's
    peak lists of every daily contact, then returns the height-weighted mean center.
    Heights are used as weights because taller (more prominent) peaks are a
    stronger signal of habitual usage time -> they should pull more strongly.

    Parameters:
    - agent                : Agent object whose social targets are being computed
    - previous_day_contacts: dict {agent_id: [contact_id, ...]} -> yesterday's daily contact network
    - agents_by_id         : dict {agent_id: Agent} -> quick lookup for contact Agent objects
    - appliance_names      : list of str -> appliances to compute targets for

    Returns a dict {appliance_name: float or None}
    -> float = height-weighted mean peak center of all contacts for that appliance
    -> None  = no contact had any peaks for that appliance, so no social shift is applied
    """
    social_targets = {}  #will hold one entry per appliance
    contact_ids    = previous_day_contacts.get(agent.agent_id, [])  #get this agent's daily contacts from yesterday

    for appliance in appliance_names:
        centers = []  #collect all peak centers from all contacts for this appliance
        weights = []  #collect corresponding heights to use as weights

        for contact_id in contact_ids:
            contact = agents_by_id.get(contact_id)  #look up the contact Agent object
            if contact is None:
                continue  #skip if the contact ID is somehow not in the agents dict

            #use previous_peak_lists: the contact's state as of yesterday
            #this is important to prevent reading peaks that have already been shifted today
            for c, h, w in contact.previous_peak_lists.get(appliance, []):
                centers.append(c)  #record the center of this peak
                weights.append(h)  #record the height as a weight

        if centers:
            #compute a height-weighted mean: taller peaks of neighbors pull more strongly
            social_targets[appliance] = float(np.average(centers, weights=weights))
        else:
            social_targets[appliance] = None  #no neighbor data available, no social shift

    return social_targets


#-----------------------------------------------------------------------
#Individual (per-agent) metrics
#-----------------------------------------------------------------------

def compile_agent_day_metrics(agent, day, load, prices_96):
    """
    Compute all per-agent metrics for a single simulated day.

    Parameters:
    - agent     : Agent object (post-load, post-shift for this day)
    - day       : int -> 0-indexed day number
    - load      : np.array of shape (96,) -> agent's 15-min kW load profile for this day
    - prices_96 : np.array of shape (96,) -> price per 15-min slot (each hour price repeated 4 times)

    Returns a dict of scalar metrics, one row destined for df_agent_daily
    """
    baseline_kw = agent.appliance_chars["Baseline"]["power_kw"]  #this agent's always-on power draw

    #remove always-on baseline from the load to isolate behaviorally driven appliance consumption
    #clip to zero to handle minor negative values from sampling variance
    appliance_load = np.maximum(load - baseline_kw, 0.0)

    #--- Energy ---
    #multiply by 0.25 because each slot is 15 minutes = 0.25 hours
    total_appliance_kwh = float(appliance_load.sum() * 0.25)

    #--- Cost ---
    #raw cost: sum of (appliance kW * price * 0.25h) across all 96 slots
    raw_cost = float((appliance_load * prices_96).sum() * 0.25)

    #normalized cost: pricing units per kWh of appliance load
    #dividing by energy makes the cost comparable across agents regardless of total consumption
    if total_appliance_kwh > 0.0:
        normalized_cost = raw_cost / total_appliance_kwh
    else:
        normalized_cost = 0.0  #avoid division by zero for agents with no appliance use today

    #--- Flexibility ---
    #these are the shift magnitudes stored on the agent by apply_shifts() earlier this day
    #on day 0 they are all 0.0 because no shifting has been applied yet
    total_flex  = agent.last_total_flexibility
    price_flex  = agent.last_price_flexibility
    social_flex = agent.last_social_flexibility

    #--- Discomfort ---
    #cumulative drift of peak centers away from the agent's own day-0 preferred positions
    discomfort = agent.compute_discomfort()

    #--- Cost-flexibility ratio ---
    #higher ratio = more costly per unit of flexibility provided to the grid
    #undefined (NaN) on day 0 because flexibility is 0 -> ratio would be division by zero
    if total_flex > 0.0:
        cost_flex_ratio = normalized_cost / total_flex
    else:
        cost_flex_ratio = float("nan")

    return {
        "day":                        day,
        "agent_id":                   agent.agent_id,
        "dominant_group":             agent.dominant_group,
        "habit_str":                  agent.habit_str,
        "price_sens":                 agent.price_sens,
        "soc_suc":                    agent.soc_suc,
        "individual_flexibility":     total_flex,
        "price_shift_contribution":   price_flex,
        "social_shift_contribution":  social_flex,
        "individual_cost_raw":        raw_cost,
        "individual_cost_normalized": normalized_cost,
        "individual_discomfort":      discomfort,
        "total_appliance_kwh":        total_appliance_kwh,
        "cost_flex_ratio":            cost_flex_ratio,
    }


#-----------------------------------------------------------------------
#System-level (per-day) metrics
#-----------------------------------------------------------------------

def gini_coefficient(load_array):
    """
    Compute the Gini coefficient of a 96-slot load profile.

    Gini is in the range [0, 1]:
    -> 0 = perfectly flat load (all 96 slots draw exactly the same power)
    -> 1 = all load concentrated in a single slot (maximally unequal)
    Used as a measure of temporal load inequality which is relevant for grid stress.
    A spiky post-shift rebound would push the Gini toward 1.
    """
    arr = np.sort(np.abs(load_array).ravel())  #sort absolute values from low to high
    n   = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0  #empty or flat array, no inequality
    index = np.arange(1, n + 1)  #rank index 1 to n
    #standard Gini formula on sorted array
    return float((2.0 * (index * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


def count_rebound_peaks(aggregate_96, prominence):
    """
    Count the number of prominent local maxima in the aggregate 96-slot load profile.

    Uses scipy.signal.find_peaks with a prominence filter so that minor bumps
    (e.g. a single slot slightly higher than its neighbors) are excluded.
    The prominence threshold is expressed as a fraction of the day's mean load,
    making it scale-invariant with respect to the number of agents.

    Parameters:
    - aggregate_96 : np.array of shape (96,) -> system aggregate load in kW per 15-min slot
    - prominence   : float -> minimum peak prominence as a fraction of mean load
        -> e.g. 0.5 means a peak must stand at least 50% of mean load above its surroundings

    Returns an int: the number of prominent peaks detected on this day
    """
    mean_load      = aggregate_96.mean()
    abs_prominence = prominence * mean_load  #convert fractional threshold to absolute kW
    peaks, _       = find_peaks(aggregate_96, prominence=abs_prominence)  #detect prominent maxima
    return int(len(peaks))


def compile_day_metrics(day, aggregate_96, prices_24h, agent_records, prominence):
    """
    Compute all system-level metrics for a single simulated day.

    Parameters:
    - day          : int -> 0-indexed day number
    - aggregate_96 : np.array of shape (96,) -> total load across all agents in kW per 15-min slot
    - prices_24h   : list or array of 24 floats -> hourly prices used on this day
    - agent_records: list of dicts -> output of compile_agent_day_metrics for all agents this day
    - prominence   : float -> prominence threshold passed to count_rebound_peaks

    Returns a dict of scalar metrics, one row destined for df_daily
    """
    prices_arr = np.array(prices_24h)  #convert to numpy for vectorized operations

    #--- Load shape metrics ---
    peak_load        = float(aggregate_96.max())  #single highest 15-min demand of the day in kW
    mean_load        = float(aggregate_96.mean()) #average 15-min demand across the day in kW
    par              = (peak_load / mean_load) if mean_load > 0 else 0.0  #Peak-to-Average Ratio: higher = more grid stress
    load_var         = float(aggregate_96.var())  #variance of the 96-slot profile
    load_cv          = float(aggregate_96.std() / mean_load) if mean_load > 0 else 0.0  #coefficient of variation: normalized spread
    load_gini        = gini_coefficient(aggregate_96)  #temporal load inequality
    total_energy_kwh = float(aggregate_96.sum() * 0.25)  #total energy in kWh for this day
    n_peaks          = count_rebound_peaks(aggregate_96, prominence)  #number of prominent peaks (rebound detection)

    #--- Price metrics ---
    mean_price  = float(prices_arr.mean())           #average price across the day
    price_min   = float(prices_arr.min())            #cheapest hour of the day
    price_max   = float(prices_arr.max())            #most expensive hour of the day
    price_range = price_max - price_min              #spread of prices, larger = stronger shifting incentive

    #--- System-level behavioral metrics from individual agent records ---
    df_agents         = pd.DataFrame(agent_records)  #convert list of dicts to DataFrame for easy aggregation
    accum_flexibility = float(df_agents["individual_flexibility"].sum())         #sum of all agents' shifts today
    accum_cost_norm   = float(df_agents["individual_cost_normalized"].mean())    #mean normalized cost across all agents
    mean_discomfort   = float(df_agents["individual_discomfort"].mean())         #mean cumulative discomfort across all agents

    #--- Per-dominant-group metrics ---
    #directly addresses RQ1 at the system level by breaking out each behavioral group separately
    group_stats = {}
    for group in ["Habit-driven", "Price-responsive", "Social-influenced"]:
        subset   = df_agents[df_agents["dominant_group"] == group]  #filter to only this group's rows
        safe_key = group.lower().replace("-", "_").replace(" ", "_")  #turn "Habit-driven" into "habit_driven" for column names
        if len(subset) > 0:
            group_stats[f"flex_{safe_key}_mean"]              = float(subset["individual_flexibility"].mean())
            group_stats[f"cost_norm_{safe_key}_mean"]         = float(subset["individual_cost_normalized"].mean())
            group_stats[f"discomfort_{safe_key}_mean"]        = float(subset["individual_discomfort"].mean())
            group_stats[f"cost_flex_ratio_{safe_key}_mean"]   = float(subset["cost_flex_ratio"].mean(skipna=True))  #skipna for day-0 NaN
            group_stats[f"n_{safe_key}"]                      = int(len(subset))
        else:
            #group not present in this run, fill with NaN to avoid KeyErrors downstream
            group_stats[f"flex_{safe_key}_mean"]              = float("nan")
            group_stats[f"cost_norm_{safe_key}_mean"]         = float("nan")
            group_stats[f"discomfort_{safe_key}_mean"]        = float("nan")
            group_stats[f"cost_flex_ratio_{safe_key}_mean"]   = float("nan")
            group_stats[f"n_{safe_key}"]                      = 0

    #assemble the final record dict for this day
    record = {
        "day":                           day,
        #load shape
        "peak_load_kw":                  peak_load,
        "mean_load_kw":                  mean_load,
        "par":                           par,
        "load_variance":                 load_var,
        "load_cv":                       load_cv,
        "load_gini":                     load_gini,
        "total_energy_kwh":              total_energy_kwh,
        "n_rebound_peaks":               n_peaks,
        #price
        "mean_price":                    mean_price,
        "price_min":                     price_min,
        "price_max":                     price_max,
        "price_range":                   price_range,
        #aggregate behavioral
        "accumulative_flexibility":      accum_flexibility,
        "accumulative_cost_norm_mean":   accum_cost_norm,
        "mean_discomfort":               mean_discomfort,
    }
    record.update(group_stats)  #merge in the per-group stats
    return record


#-----------------------------------------------------------------------
#DataFrame assembly
#-----------------------------------------------------------------------

def build_dataframes(agent_day_records, day_records):
    """
    Convert the raw list-of-dicts records collected during simulation into
    two analysis-ready pandas DataFrames.

    Also adds a derived cumulative column to df_agent_daily:
    -> cumulative_flexibility: running total of individual_flexibility per agent across days
       (individual_discomfort is already cumulative by design, measuring drift from day 0)

    Parameters:
    - agent_day_records : list of dicts from compile_agent_day_metrics (all agents, all days)
    - day_records       : list of dicts from compile_day_metrics (one per day)

    Returns:
    - df_agent_daily : pd.DataFrame with one row per (agent, day)
    - df_daily       : pd.DataFrame with one row per day
    """
    df_agent_daily = pd.DataFrame(agent_day_records)  #one row per (agent, day)
    df_daily       = pd.DataFrame(day_records)         #one row per day

    #sort for predictable ordering in analysis
    df_agent_daily = df_agent_daily.sort_values(["day", "agent_id"]).reset_index(drop=True)
    df_daily       = df_daily.sort_values("day").reset_index(drop=True)

    #add cumulative flexibility per agent over the simulation period
    #useful for tracking the total grid service each household has provided
    df_agent_daily["cumulative_flexibility"] = (
        df_agent_daily
        .sort_values("day")
        .groupby("agent_id")["individual_flexibility"]
        .cumsum()
        .values
    )

    return df_agent_daily, df_daily