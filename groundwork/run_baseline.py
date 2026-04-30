import numpy as np
import matplotlib.pyplot as plt


"""
This code generates a daily electricity load profile for every agent at a 15-minute granularity
-> It determines, when, how often, how long, how intense (appliance variant power draw) appliances are ran

It includes function run_simulation, giving access to baselines, this code was changed in the final, shiftable full program, but logic remained
Used to set baselines for the price model

The core methodology is adopted from Williams et al. (2025) 
For their implementation, refer to: https://github.com/alikazemian-bot/AMPED-Residential/blob/main/AMPED-Residential%20Agent-based%20Model%20for%20Predicting%20Electricity%20Demand.ipynb

Methodology:
    - Each appliance has a 24h probability distribution -> determines WHEN it is likely to be switched on
    - The number of daily uses is determined by iterating over the switch-on data sourced from Yilmaz et al. (2017), which draws on the
      UK Household electricity survey. On every value, a random float is drawn between 0-1, if this number exceeds the value, the times this appliance will be used on this day will go +1
      The outcome is independent of previous trials.
    - Once a start-hour is sampled from the distribution, a random quarter within that hour
      (:00, :15, :30, :45) is added to get a 15-minute resolution start time
    - Intensity and length are sampled once per agent at initialisation and stay fixed,
      using the characteristics sourced from the code by Williams et al. (2025)
    - A flat baseline load represents always-on devices (e.g. fridge, router, standby). This load is evenly spread over the day (doensn't include rythmic cycling like in Williams et al. (2025)
      implementation because that is not behavior influences. Size of the baseline is sampled per agent and was set by combining Williams et al. (2025) and Liander aggregate data of a residential area.

"""



#Same data as in the baseline_distributions file, sourced from Yilmaz et al. (2017)
#Each value is a switch on probability (for each hour, but I will only use them to determine the amount of activations, hours are determined by agents' own distributions

data = {
    "Dishwasher":   [0.016,0.0071,0.0071,0.0044,0.0044,0.0017,0.008,0.0266,0.038,0.032,0.03,0.021,0.024,0.034,0.025,0.025,0.0196,0.027,0.0577,0.06577,0.0524,0.0328,0.0488,0.0355],
    "Washing":      [0.00525,0.00617,0.00795,0.00527,0.00354,0.01777,0.03021,0.11463,0.15107,0.2061,0.2213,0.1812,0.1412,0.12976,0.0871,0.08,0.07466,0.0879,0.0613,0.05594,0.05596,0.038221,0.03464,0.01422],
    "Tumble_Drier": [0.004,0,0,0,0.00266,0.00177,0.01155,0.0231,0.0311,0.0391,0.0462,0.0524,0.0506,0.03,0.02,0.03,0.025,0.02,0.0453,0.0435,0.0426,0.0293,0.0204,0.0088],
    "Cooker":       [0.00259,0.00259,0.0103,0.01168,0.01168,0.02727,0.048,0.11,0.1221,0.111,0.1051,0.1233,0.148,0.05,0.04,0.03,0.08,0.1,0.15,0.1519,0.0883,0.0494,0.0337,0.01168],
    "Oven":         [0.00259,0.00519,0.00649,0.00259,0.00519,0.00519,0.0337,0.0688,0.0467,0.0337,0.03636,0.0415,0.0688,0.018,0.014,0.02,0.02,0.014,0.0987,0.0753,0.03636,0.0259,0.035,0.0324],
    "Grill":        [0,0,0.0012,0.0026,0.002597,0.00129,0.0012,0.0091,0.00779,0.00779,0.00259,0.01558,0.0519,0.01,0.002,0.001,0.025,0.03,0.0311,0.01558,0.0103,0.0011,0.00259,0.0026],
    "Hob":          [0.0025,0.0013,0.0026,0.00389,0.001298,0.00389,0.0155,0.0909,0.112,0.0688,0.061,0.0493,0.0727,0.01,0.03,0.02,0.04,0.08,0.13,0.089,0.0584,0.04025,0.01039,0.00779],
    "TV":           [0.0497,0.0135,0.01917,0.07123,0.042831,0.07345,0.1345,0.2,0.23,0.2228,0.17986,0.08254,0.12102,0.117024,0.0817,0.11195,0.168505,0.18,0.210705,0.36,0.35,0.31,0.3078,0.0904],
    "Electronics":  [0.02485,0.00675,0.009585,0.035615,0.0214155,0.036725,0.06725,0.1,0.115,0.1114,0.08993,0.04127,0.06051,0.058512,0.04085,0.055975,0.0842525,0.09,0.1053525,0.18,0.175,0.155,0.1539,0.0452],
}


#Appliance characteristics from Williams et al. (2025), refer to the Github link at the top of this file for source
#Power in kW and runtime in minutes are sampled once per agent and remain fixed from normal distributions with these parameters
#Reflects different households own different appliance models with different efficiencies
#Max_uses_mu and max_uses_sigma added to cap physically unrealistic daily use counts (occupancy filter)

characteristics = {
    "Dishwasher": {"power_mu": 0.65, "power_sigma": 0.20, "runtime_mu": 60,  "runtime_sigma": 15, "max_uses_mu": 1.0, "max_uses_sigma": 0.5},
    "Washing": {"power_mu": 0.65, "power_sigma": 0.40, "runtime_mu": 45,  "runtime_sigma": 15, "max_uses_mu": 1.5, "max_uses_sigma": 0.5},
    "Tumble_Drier": {"power_mu": 1.10, "power_sigma": 0.70, "runtime_mu": 60,  "runtime_sigma": 15, "max_uses_mu": 1.5, "max_uses_sigma": 0.5},
    "Cooker": {"power_mu": 1.00, "power_sigma": 0.80, "runtime_mu": 30,  "runtime_sigma": 15, "max_uses_mu": 4.0, "max_uses_sigma": 0.5},
    "Oven": {"power_mu": 0.70, "power_sigma": 0.50, "runtime_mu": 30,  "runtime_sigma": 15, "max_uses_mu": 2.0, "max_uses_sigma": 0.5},
    "Grill": {"power_mu": 1.50, "power_sigma": 0.50, "runtime_mu": 20,  "runtime_sigma": 10, "max_uses_mu": 3.0, "max_uses_sigma": 1.0},
    "Hob": {"power_mu": 1.00, "power_sigma": 0.80, "runtime_mu": 20,  "runtime_sigma": 10, "max_uses_mu": 4.0, "max_uses_sigma": 1.0},
    "TV": {"power_mu": 0.10, "power_sigma": 0.10, "runtime_mu": 150, "runtime_sigma": 60, "max_uses_mu": 6.0, "max_uses_sigma": 3.0},
    "Electronics": {"power_mu": 0.80, "power_sigma": 0.50, "runtime_mu": 30,  "runtime_sigma": 60, "max_uses_mu": 8.0, "max_uses_sigma": 5.0},
}


def multi_peak_distribution(peak_list, baseline_probability=0.005): #Exact same function as from baseline_distributions, turns peak list into valid prob. distribution
    """
    Builds a final 24-hour probability distribution as a sum of Gaussians.

    Each entry in peak_list is a tuple representing a peak: (center_hour, height, width).
      - center_hour: on what hour the peak is located
      - height: peak magnitude before normalization
      - width: standard deviation in hours, if larger = broader bump

    A small flat baseline is added so every hour has at least some probability,
    to account for schedule irregularities.
    The result is then divided by the lists' sum for all elements so it adds up to 1.
    """
    distribution = np.zeros(24) #initialize where the final distribution will be stored

    for center_hour, height, width in peak_list:
        gaussian = height * np.exp(-(np.arange(24, dtype=float) - center_hour)**2 / (2 * width**2)) #compute gaussians for all peaks
        distribution = distribution + gaussian #use distribution for cumulative storage of all gaussian data

    distribution += baseline_probability
    distribution /= distribution.sum() #normalize to a legal probability distribution which sums to 1.
    return distribution

#The adjust peaks function has been implemented to slighly shift all baseline distributions
#This change was made because of:
#1. A mismatch of peaking and minimal usage hours against real DSO data: without adjustments the model peaks at 12, with adjustments, this peak is shifted to the morning and mirrors a similar structure to general electricity usage curves 
#2. This mismatch would make the usage of baseline EPEX pricing illogical

def adjust_peaks(peak_list):
    adjusted = []
    for center, height, width in peak_list:
        if 8 <= center <= 13:
            center = center - 2.8
            height = height + 0.1   
            width = max(0.1, width - 0.3) 
        if 18 <= center <= 22:
            center = center - 2
            height = height + 0.5 
            width = width + 0.5
        adjusted.append((center, height, width))   

    return adjusted
    
#Fitted Gaussian peak parameters from the baseline_distributions script, copy pasted values WITH adjusted_peaks
#Each tuple is (center_hour, height, width) for one Gaussian peak
dishwasher_baseline = multi_peak_distribution(adjust_peaks([(18.956315861064553, 0.11089493744353027, 1.357159179717805), (22.36027871623913, 0.07797395460554271, 0.771610889785124), (8.342960587362622, 0.05962600655372998, 1.4002245664011947), (13.314358999677186, 0.05, 1.8967520713194168)]))

washing_baseline = multi_peak_distribution(adjust_peaks([(9.545167198482746, 0.7790387292870447, 2.0244500172858944), (15.60115790675859, 0.33413792870222453, 4.41870480568728)]))

tumble_drier_baseline = multi_peak_distribution(adjust_peaks([(11.62212340248307, 1.208933954615057, 1.417355530924213), (15.14387596374066, 0.5874527036199261, 0.6935500197957881), (8.466416528180892, 0.8074778473489305, 1.6964259012427485), (19.228974958574852, 1.1762361135932435, 2.0189138827026043)]))

cooker_baseline = multi_peak_distribution(adjust_peaks([(18.153061228523544, 1.2582844975217007, 2.3746642890017986), (18.59097312234728, 0.6816961836975054, 0.6823292016136122), (11.795692959571095, 1.3953243916233937, 0.7497850471616758), (8.425787475465675, 1.4349642274396488, 2.0156280604983725)]))

oven_baseline = multi_peak_distribution(adjust_peaks([(18.458235649470627, 0.37922201728501226, 0.3046711094792303), (11.738449852884226, 0.10562985711043978, 0.693090134001847), (7.545761516821308, 0.09437391474553104, 1.4517620706320356), (21.682585961077926, 0.05, 5.0)]))

grill_baseline = multi_peak_distribution(adjust_peaks([(11.924944463813114, 0.2809556783426315, 0.5815431599228617), (7.95645163258788, 0.05, 1.1849438827408596), (17.781582463893, 0.16119629241571146, 1.2220586595698648), (16.406340924304615, 0.18506115113627022, 0.3062559743946323)]))

hob_baseline = multi_peak_distribution(adjust_peaks([(18.22784853881954, 0.15979319165991923, 1.5625426291399946), (7.638187212842916, 0.14974437128118528, 0.8485970203176886), (10.844037221309907, 0.08465959403479242, 1.7520044963356904)]))

tv_baseline = multi_peak_distribution(adjust_peaks([(8.178903464086682, 2.4674115102346237, 2.509022410606133), (19.505228878954412, 3.575007824823448, 0.4148362459235295), (18.615440531185566, 2.3700823150476538, 3.4375909799857913), (21.537457239174596, 3.6798085662182154, 0.4212647664517777)]))

electronics_baseline = multi_peak_distribution(adjust_peaks([(8.178903464086682, 2.4674115102346237, 2.509022410606133), (19.505228878954412, 3.575007824823448, 0.4148362459235295), (18.615440531185566, 2.3700823150476538, 3.4375909799857913), (21.537457239174596, 3.6798085662182154, 0.4212647664517777)]))

ev_baseline = multi_peak_distribution(adjust_peaks([(0.6, 2.2, 1.8), (14.0, 0.5, 3), (19.5, 2.5, 2.3), (24, 1, 2.7)]))

#EVs handled separately because it uses a single daily draw 
baselines = {
    "Dishwasher": dishwasher_baseline,
    "Washing": washing_baseline,
    "Tumble_Drier": tumble_drier_baseline,
    "Cooker": cooker_baseline,
    "Oven": oven_baseline,
    "Grill": grill_baseline,
    "Hob": hob_baseline,
    "TV": tv_baseline,
    "Electronics": electronics_baseline,
}


#-----------------------------
#Initialising an agent
#-----------------------------

def sample_agent_appliances(random_state):
    """
    Setting the fixed characteristics for one agent per appliance, refer to top for data source
    Returns a dictonary with power + runtime for every agent and the has_ev boolean
    """
    agent_appliances = {}
    for name, chara in characteristics.items():
        power = abs(random_state.normal(chara["power_mu"], chara["power_sigma"])) #abs as safety for extreme outliers
        runtime = abs(random_state.normal(chara["runtime_mu"], chara["runtime_sigma"]))
        runtime = max(1, round(runtime)) #again, safety for extreme outliers
        #Max uses sampled per agent so households differ in how intensively they use each appliance
        max_uses = abs(random_state.normal(chara["max_uses_mu"], chara["max_uses_sigma"]))
        max_uses = max(1, round(max_uses)) #at least 1, must be integer
        agent_appliances[name] = {
            "power_kw": power,
            "runtime_min": runtime,
            "max_uses": max_uses}
   
    #For standby, fridge etc:
    baseline_power = abs(random_state.normal(0.4, 0.15)) #sourced from Williams et al. (2025)
    agent_appliances["Baseline"] = {"power_kw": baseline_power}

    #Setting up EV's 
    #EV characteristics from Robinson et al. (2013) Table 4

    has_ev = random_state.random() < 0.05 #5% EV ownership is a hyperparameter, to be adjusted when adressing specific countries
    if has_ev:
        ev_power   = abs(random_state.normal(3.3, 0.3)) #3 kv mentioned in Robinson et al., converted to 3.3 kw
        ev_runtime = abs(random_state.normal(3.1 * 60, 20)) # 3.1 hours converted to minutes, 20 is a self set hyperparameter to account for variance in batteries/car-types
                                                                                                               
        ev_runtime = max(15, round(ev_runtime))  #safety for edge cases
        agent_appliances["EV"] = {
            "power_kw": ev_power,
            "runtime_min": ev_runtime
        }
    return agent_appliances, has_ev


#--------------------------
#Building the daily usage
#--------------------------

def build_daily_load(agent_appliances, has_ev, random_state, previous_overflow=None):
    """
    Generate one day electricity load profile for single agent.

    Overnight overflow from the previous day is added at the top so that
    appliances that started near midnight day-1 contribute to day.

    Parameters:
    - agent_appliances: dictionary from sample_agent_appliances()
    - has_ev: bool, True if possesses electric vehicle 
    - random_state 
    - previous_overflow: array of shape (96,) or None, overflow load carried over from the previous day.

    Returns:
    - load_profile: kW per 15-min slot for today
    - overflow: kW per 15-min slot spilling to tomorrow
    - schedule: dictionary: {appliance_name: [start_slot, ...]}
    """

    load_profile = np.zeros(96) #24h x 4 for 15 min intervals
    overflow = np.zeros(96)   
    schedule = {}

    #Add day-1 overload to today right away
    if previous_overflow is not None:
        for slot in range(96):
            load_profile[slot] = load_profile[slot] + previous_overflow[slot]

    #Spread baseline draw across every slot of the day
    baseline_power = agent_appliances["Baseline"]["power_kw"]
    for slot in range(96):
        load_profile[slot] = load_profile[slot] + baseline_power

    #Now for normal appliances
    for name, probs in data.items():  #iterate over all switch on data, if it hits, n+1 usage
        n_uses = 0
        for p in probs:
            draw = random_state.random()   
            if draw < p:
                n_uses = n_uses + 1        #this hour produced a switch-on event

        #Cap n_uses at the agent's sampled max_uses for this appliance
        #Prevents physically impossible use counts (e.g. TV 8 times in one day)
        #regardless of how the distribution shifts under price or social influence
        n_uses = min(n_uses, agent_appliances[name]["max_uses"])

        #Skip this appliance if not used today
        if n_uses == 0:
            schedule[name] = []
            continue

        #Now sample which hour to take
        dist = baselines[name].copy()             
        start_hours = random_state.choice(np.arange(24), size=n_uses, p=dist)

        #Let agent sample 5 times again for new hours, if an hour was already chosen to prevent lot of duplicate use and create more logical, over-the-day behavior
        used_hours = []
        for hour in start_hours:
            for i in range(5):
                if hour not in used_hours:
                    used_hours.append(hour)
                    break
                hour = random_state.choice(np.arange(24), p=dist)
            else:
            #after 5 failed attempts, take the value anyway
                used_hours.append(hour)

        
        #Get own characteristic values for this appliance
        power_kw = agent_appliances[name]["power_kw"]
        runtime_min = agent_appliances[name]["runtime_min"]
        n_slots = max(1, round(runtime_min / 15))  #runtime in slots

        start_slots = []
        for h in used_hours:
            #To determine precise 15-minute start slot
            #pick a random quarter-hour within the sampled hour
            quarter = int(random_state.integers(0, 4))
            start_slot = h * 4 + quarter    #absolute slot index within today (0-95)



            #Spread load over each slot of the runtime.
            #If slot exceeds 95 , place load in the overflow array
            for slot in range(n_slots):
                target_slot = start_slot + slot

                if target_slot < 96:
                    #today
                    load_profile[target_slot] = load_profile[target_slot] + power_kw
                else:
                    #overflow
                    overflow_slot = target_slot - 96
                    overflow[overflow_slot] = overflow[overflow_slot] + power_kw

            start_slots.append(start_slot)

        schedule[name] = start_slots

    #handling EVs
    #If the probability hits, one start-time is sampled from the EV MPD,
    #and load is spread over the charging duration exactly as done above
    if has_ev:
        ev_draw = random_state.random()
        if ev_draw < 41.6 / 180: #taken from table 4 (private home value divided by 6 months)
            #sample start-hour from the EV evening-peak distribution
            start_hour = int(random_state.choice(np.arange(24), p=ev_baseline))
            quarter = int(random_state.integers(0, 4))
            start_slot = start_hour * 4 + quarter

            power_kw = agent_appliances["EV"]["power_kw"]
            runtime_min = agent_appliances["EV"]["runtime_min"]
            n_slots = max(1, round(runtime_min / 15))

            ev_slots = []
            for s in range(n_slots):
                target_slot = start_slot + s

                if target_slot < 96:
                    load_profile[target_slot] = load_profile[target_slot] + power_kw
                else:
                    overflow_slot = target_slot - 96
                    if overflow_slot < 96:
                        overflow[overflow_slot] = overflow[overflow_slot] + power_kw

                ev_slots.append(start_slot + s)
            schedule["EV"] = ev_slots
        else:
            schedule["EV"] = []  # no charging today
    return load_profile, overflow, schedule


#------------------
#Simulation runner
#------------------

def run_simulation(days=7, random_state=2, agents=150, plots=None, median_plot=True):
    """
    Run the full simulation for a given number of days.
    Parameters:
    - days: number of days to simulate
    - random_state: integer seed for the master RNG, change this to get different runs
                    agent seeds are derived from it so the same value always gives identical output
    - agents: number of household agents
    - plots: list of day indices (starting from 0 (day one)) to plot aggregate load for, e.g. [0, 6, 13]
             if None, no plots are shown
    - median_plot: Boolean, True if you want to print a median plot of the entire period
    
    Returns:
    - all_aggregates: each entry is a day with a 96-slot aggregate load array
    - all_daily_profiles: each entry is a day with a list of per-agent 96-slot arrays
    """
     
    #Changing random_state gives a completely different run, same value always reproduces
    rng = np.random.default_rng(seed=random_state)
    agent_seeds = rng.integers(0, 1000000, size=agents)

    #Create one independent RNG per agent from the derived seeds
    agent_random_states = []
    for i in range(agents):
        agent_random_states.append(np.random.default_rng(seed=int(agent_seeds[i])))

    #Initialise appliance hardware once, stays fixed for all days
    agent_appliance_sets = []
    agent_has_ev = []
    for random_state_i in agent_random_states:
        appliances, has_ev = sample_agent_appliances(random_state_i)
        agent_appliance_sets.append(appliances)
        agent_has_ev.append(has_ev)

    #Storage for results across all days
    all_aggregates = []
    all_daily_profiles = []

    #Overflows start empty on day 0
    current_overflows = []
    for i in range(agents):
        current_overflows.append(None)

    #Main simulation loop
    for day in range(days):
        day_profiles = []
        next_overflows = []

        for i in range(agents):
            load, overflow, sched = build_daily_load(
                agent_appliances = agent_appliance_sets[i],
                has_ev = agent_has_ev[i],
                random_state = agent_random_states[i],
                previous_overflow = current_overflows[i])
            day_profiles.append(load)
            next_overflows.append(overflow)

        #Carry overflows forward to the next day
        current_overflows = next_overflows

        #Aggregate across all agents for this day
        aggregate = np.zeros(96)
        for profile in day_profiles:
            aggregate = aggregate + profile

        all_aggregates.append(aggregate)
        all_daily_profiles.append(day_profiles)

        print(f"Day {day + 1}/{days} done  |  peak load: {aggregate.max():.2f} kW  |  total: {aggregate.sum() * 0.25:.1f} kW")

    #plotting requested days
    if plots is not None:
        time_axis = np.linspace(0, 24, 96, endpoint=False)

        for day_index in plots:
            if day_index >= days:
                print(f"Warning: day index {day_index} requested but only {days} days were simulated, skipped")
                continue

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.fill_between(time_axis, all_aggregates[day_index], color="salmon", alpha=0.7)
            ax.set_ylabel("kW")
            ax.set_xlabel("Hour of day")
            ax.set_title(f"Aggregate load of Day {day_index + 1} ({agents} agents on random state {random_state})")
            ax.grid(alpha=0.3)
            plt.xticks(range(25))
            plt.tight_layout()
            plt.show()
    
    if median_plot: 
        time_axis = np.linspace(0, 24, 96, endpoint=False)
        aggregates_array = np.array(all_aggregates)
        median_profile = np.median(aggregates_array, axis=0)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(time_axis, median_profile, color="lightblue", alpha=0.7)
        ax.set_ylabel("kW")
        ax.set_xlabel("Hour of day")
        ax.set_title(f"Median aggregate load profile ({days} days, {agents} agents, seed {random_state})")
        ax.grid(alpha=0.3)
        plt.xticks(range(25))
        plt.tight_layout()
        plt.show()
 
    return all_aggregates, all_daily_profiles

#------------------
#Example call
#------------------

results, profiles = run_simulation(days=5, random_state=100, agents=1000, median_plot=True)
