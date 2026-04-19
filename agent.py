import numpy as np
from load_profile import baseline_peak_tuples, multi_peak_distribution, sample_agent_appliances

"""
Each Agent object represents one household in the simulation. It has:
    -> Behavioral parameters (habit_str, price_sens, soc_suc) sampled from the
       beta distributions in Setting_Parameters.py
    -> Fixed appliance characteristics (power draw, runtime, max daily uses) sampled once
       at initialisation and remain static
    -> Personal peak lists: one list of (center, height, width) up to 4 peaks per appliance (as determined by baseline_distributions.py)
    -> Probability distributions derived from those peak lists
       -> Passed to load_profile.build_daily_load() every simulated day

Shifting order each day:
    -> Day 0 only: habit shift is applied once at init to set initial preferred times
    -> Day 1+: price shift moves peaks toward the nearest cheap hour
    -> Day 1+: social shift moves peaks toward the weighted mean of yesterday's contacts
    -> Heights and widths are never changed after day 0 initialization
"""

#These scaling factors control how large each behavioral shift is
#They are used as defaults in run_model.py and can be overridden there as parameter, used to tweak the model in development and gives some more flexibility for generalisation

default_epsilon_habit = 1.8   #magnitude height/width changed at initialization (one time only)
default_epsilon_price = 0.4   #how strongly price signals pull peak centers toward cheap hours (per day)
default_epsilon_social = 0.4  #how strongly social contact pulls peak centers toward neighbors (per day)


#---------------------------------------------------------------------------------
#Appliance shifting rates
#Per-appliance flexibility in the range [0, 1]
#Higher value = this appliance responds more strongly to price and social signals
#These multiply the epsilon values so the shift per appliance is:
#   shift = behavioral_param * appliance_rate * epsilon * (target - center)
#---------------------------------------------------------------------------------
appliance_shift_rates = {
    #Highest flexibility willingness -> Based on Berg et al. (2024) 
    "Washing": 0.75,  
    "Dishwasher": 0.75, 
    "Tumble_Drier": 0.60,  

    #Medium flexibility (behavioral/leisure)
    "Electronics": 0.30, #moderate flexibility, general-purpose usage patterns
    "TV": 0.30, #leisure-driven, light flexibility

    #Lowest flexibility willingness (routine bound) -> Based on Berg et al. (2024)
    "Oven": 0.15,  
    "Cooker": 0.15,  
    "Hob": 0.15,  
    "Grill": 0.15,  
    
    "EV": 0.35} #High future potential for flexibility, but still grounded in routine -> based on Sørensen (2021)



class Agent:
    """
    Instance of a household

    Inits:
    -> agent_id: unique ID matching entry in networks.json (e.g. AG001)
    -> dominant_group: Habit-driven, Price-responsive or Social-influenced
    -> habit_str: Habit strength drawn from Beta distribution 
    -> price_sens: Price sensitivity drawn from Beta distribution 
    -> soc_suc: Social susceptibility drawn from Beta distribution 
    -> rng: personal RNG derived from master seed in run_model.py
    -> epsilon_habit: Passed in from run_model.py so it can be tuned easily, used to init the peaks
    """

    def __init__(self, agent_id, dominant_group, habit_str, price_sens, soc_suc, rng, epsilon_habit=default_epsilon_habit):

        self.agent_id = agent_id        
        self.dominant_group = dominant_group  
        self.habit_str = float(habit_str)
        self.price_sens = float(price_sens)
        self.soc_suc = float(soc_suc)
        self.rng = rng  
        
        #sample fixed appliance hardware charcteristics once (power draw, runtime, max daily uses)
        self.appliance_chars, self.has_ev = sample_agent_appliances(rng)

        #personal Gaussian peak lists, one list of (center, height, width) tuples per appliance
        #current_peak_lists = peaks used for today's load simulation
        #previous_peak_lists = snapshot saved at the start of each day before shifting
        self.current_peak_lists = {}   #appliance_name = [(center, height, width), for amount of peaks] for all appliances
        self.previous_peak_lists = {} #for reference with social shift

        #probability distributions derived from current_peak_lists
        #passed as custom_distributions to build_daily_load() each day
        self.current_distributions = {}

        #overflow load carried over from the previous day
        self.previous_overflow = None

        #shift magnitude trackers for metrics
        self.last_total_flexibility = 0.0  #total absolute peak center shift across all appliances today
        self.last_price_flexibility = 0.0  #portion of the total shift caused by the price signal
        self.last_social_flexibility = 0.0  #portion of the total shift caused by the social signal

        #apply the one-time habit shift and set up initial peak lists and distributions
        self.initialize_peaks(epsilon_habit)

    def initialize_peaks(self, epsilon_habit):
        """
        Set up personal peak lists at the start with habit parameter
        """
        for name, baseline_peaks in baseline_peak_tuples.items():
            if name == "EV" and not self.has_ev:  #skip EV peaks for agents without an EV
                continue
            habit_adjusted = []  #will hold the habit-shifted version of peaks for the appliance
            for center, height, width in baseline_peaks:
                new_height = height + self.habit_str * epsilon_habit  #raise height by habit strength
                new_width = max(0.2, width - self.habit_str * 0.5)  #habitual agents have tighter peaks, floor at 0.2
                habit_adjusted.append((center, new_height, new_width)) #center unchanged
            self.current_peak_lists[name] = habit_adjusted  #store the adjusted peaks for this appliance

        #initialize previous_peak_lists as an identical copy of current for reference after day0
        self.previous_peak_lists = {}
        for k, v in self.current_peak_lists.items():
            self.previous_peak_lists[k] = list(v)
        
        #compute probability distributions for day 0 from the habit-adjusted peaks
        self.current_distributions = {}
        for name, peaks in self.current_peak_lists.items():
            distribution = multi_peak_distribution(peaks)
            self.current_distributions[name] = distribution

        #Store day-0 peak centers for the flexibility metric
        self.initial_peak_centers = {}
        for name, peaks in self.current_peak_lists.items():
            centers_list = []
        for c, h, w in peaks:
            centers_list.append(c)
        self.initial_peak_centers[name] = centers_list


    def apply_shifts(self, price_minima, social_targets, epsilon_price, epsilon_social):
        """
        Apply price and social shifts to update this agent's peak lists for the current day.

        Shift formula applied to each peak center:
            price_delta  = price_sens * appliance_rate * epsilon_price  * (nearest_cheap_hour - center)
            social_delta = soc_suc    * appliance_rate * epsilon_social * (neighbor_mean_center - center)
            new_center   = clip(center + price_delta   + social_delta, 0.0, 23.0)

        Only peak hour changes, heights and widths are fixed after initialization

        Parameters:
        -> price_minima: hour indices of local price minima in todays prices 
        -> social_targets: dict {appliance_name: float or None}
            -> mean peak center from yesterday's daily contact network
            -> None means no contacts used this appliance yesterday thus no social shift
        -> epsilon_price: can be passed in run_model but a set parameter at top of this file
        -> epsilon_social: can be passed in run_model but a set parameter at top of this file
        """
        total_flex = 0.0  #will sum up total absolute shift across all peaks and appliances
        price_flex = 0.0  #will sum up the price-driven portion of all shifts
        social_flex = 0.0  #will sum up the social-driven portion of all shifts

        new_peak_lists = {}  #holds the updated peaks for all appliances

        for name, peaks in self.previous_peak_lists.items():
            rate = appliance_shift_rates[name]  #get this appliance's flexibility rate
            
            social_target = social_targets[name]

            new_peaks = []  #will hold the updated (center, height, width) tuples for this appliance
            for center, height, width in peaks:
                #--------------------------------
                #Implementation of price shift
                #--------------------------------
                #Goal: Find local minimum through gradient-descent style, slide down the slope
                #find the price minimum closest to the current peak center by minimizing the absolute distance
                nearest_min = min(price_minima, key=lambda h: abs(h - center))

                #the shift is scaled by: price sensitivity * appliance rate * epsilon * distance to target
                #larger distance = larger shift; once arrived at the minimum the shift becomes zero
                p_delta = self.price_sens * rate * epsilon_price * (nearest_min - center)

                #--------------------------------
                #Implementation of social shift
                #--------------------------------
                #Move toward the height-weighted mean peak center of yesterday's daily contacts
                #If no contacts ran this appliance yesterday, social_target is none -> no social shift
                if social_target is not None:
                    s_delta = self.soc_suc * rate * epsilon_social * (social_target - center)
                else:
                    s_delta = 0.0  #no social information available for this appliance

                #Add both deltas simultaneously and clip to keep the center within the 24-hour clock
                new_center = float(np.clip(center + p_delta + s_delta, 0.0, 23.0))

                #actual_shift is what really happened after the clip
                #unclipped p_delta and s_delta are stored separately to attribute contributions
                actual_shift = abs(new_center - center)
                total_flex += actual_shift        #how much this peak actually moved
                price_flex += abs(p_delta)        #price contribution before clipping
                social_flex += abs(s_delta)       #social contribution before clipping

                new_peaks.append((new_center, height, width))

            new_peak_lists[name] = new_peaks   

        #replace the current peak lists and distributions with the shifted versions
        self.current_peak_lists = new_peak_lists
        self.current_distributions = {}
        for name, peaks in self.current_peak_lists.items():
            #Recompute distributions from new peaks
            self.current_distributions[name] = multi_peak_distribution(peaks)

        #save shift magnitudes so metrics.py can read them without recomputing
        self.last_total_flexibility = total_flex
        self.last_price_flexibility = price_flex
        self.last_social_flexibility = social_flex


    def compute_discomfort(self):
        """
        Compute this agent's cumulative behavioral discomfort

        Returns the sum of abs(current_center - initial_center) across all peaks and appliances.
        -> A higher value means the agent is running appliances further from their preferred/initial times
        -> This is a measure for how much the price and social signals have changed normal routines
        -> It is cumulative, measures total drift from day 0, not just yesterday's shift (which is flexibility)
        """
        total = 0.0
        for name, peaks in self.current_peak_lists.items():
            initial_centers = self.initial_peak_centers.get(name, [])  #day-0 preferred positions
            for i, (center, height, width) in enumerate(peaks):
                if i < len(initial_centers):
                    total += abs(center - initial_centers[i])  #add distance from preferred center
        return total


    def __repr__(self):
        #gives a readable summary when printing an agent object
        return (f"Agent({self.agent_id}, {self.dominant_group}, "
                f"habit={self.habit_str:.2f}, price={self.price_sens:.2f}, "
                f"social={self.soc_suc:.2f})")
