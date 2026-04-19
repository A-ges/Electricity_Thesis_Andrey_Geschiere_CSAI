import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
"""
This code functions as the setup for the baseline probability format
It estimates three parameter gaussians (per peak), being: (time, magnitude/height, width).
This allows for easy interpretability of peaks during the simulation and a more straightforward shifting process.

Data adopted from Yilmaz et al. (2017): average hourly switch-on events per appliance
Sourced from: UK Household Electricity Survey 2011 which recorded the electric power demand of 251 UK homes and 5860
individual electrical appliances within those homes.

"""

#Each list has 24 values, where index 0 = 00:00-01:00 and 24 = 23:00-00:00
data = {
    "Dishwasher":[0.016,0.0071,0.0071,0.0044,0.0044,0.0017,0.008,0.0266,0.038,0.032,0.03,0.021,0.024,0.034,0.025,0.025,0.0196,0.027,0.0577,0.06577,0.0524,0.0328,0.0488,0.0355],
    "Washing":[0.00525,0.00617,0.00795,0.00527,0.00354,0.01777,0.03021,0.11463,0.15107,0.2061,0.2213,0.1812,0.1412,0.12976,0.0871,0.08,0.07466,0.0879,0.0613,0.05594,0.05596,0.038221,0.03464,0.01422],
    "Tumble Drier":[0.004,0,0,0,0.00266,0.00177,0.01155,0.0231,0.0311,0.0391,0.0462,0.0524,0.0506,0.03,0.02,0.03,0.025,0.02,0.0453,0.0435,0.0426,0.0293,0.0204,0.0088],
    "Cooker":[0.00259,0.00259,0.0103,0.01168,0.01168,0.02727,0.048,0.11,0.1221,0.111,0.1051,0.1233,0.148,0.05,0.04,0.03,0.08,0.1,0.15,0.1519,0.0883,0.0494,0.0337,0.01168],
    "Oven":[0.00259,0.00519,0.00649,0.00259,0.00519,0.00519,0.0337,0.0688,0.0467,0.0337,0.03636,0.0415,0.0688,0.018,0.014,0.02,0.02,0.014,0.0987,0.0753,0.03636,0.0259,0.035,0.0324],
    "Grill":[0,0,0.0012,0.0026,0.002597,0.00129,0.0012,0.0091,0.00779,0.00779,0.00259,0.01558,0.0519,0.01,0.002,0.001,0.025,0.03,0.0311,0.01558,0.0103,0.0011,0.00259,0.0026],
    "Hob":[0.0025,0.0013,0.0026,0.00389,0.001298,0.00389,0.0155,0.0909,0.112,0.0688,0.061,0.0493,0.0727,0.01,0.03,0.02,0.04,0.08,0.13,0.089,0.0584,0.04025,0.01039,0.00779],
    "TV Total":[0.0497,0.0135,0.01917,0.07123,0.042831,0.07345,0.1345,0.2,0.23,0.2228,0.17986,0.08254,0.12102,0.117024,0.0817,0.11195,0.168505,0.18,0.210705,0.36,0.35,0.31,0.3078,0.0904],
    "Electronics":[0.02485,0.00675,0.009585,0.035615,0.0214155,0.036725,0.06725,0.1,0.115,0.1114,0.08993,0.04127,0.06051,0.058512,0.04085,0.055975,0.0842525,0.09,0.1053525,0.18,0.175,0.155,0.1539,0.0452]
    #ELECTRONIC VEHICLE SEPERATELY ADDED AS FINISHED GAUSSIAN
}

normalized_data = {} #dictionary that holds probabilities (summing to 1)

for appliance, hourly_vals in data.items():
    vals = np.array(hourly_vals)
    probs = vals / vals.sum()
    normalized_data[appliance] = probs

#---------------------------------------------------------
# DISTRIBUTION FUNCTION (Creates the final distribution)
#---------------------------------------------------------

def multi_peak_distribution(peak_list, baseline_probability=0.005):
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


#--------------------------------------------------------------------
# OBJECTIVE FUNCTION (function to minimize during fitting peaks)
#--------------------------------------------------------------------

def sum_of_squared_errors(flat_parameters, number_of_peaks, target_probabilities):
    """
    Takes a flat array of parameters, returns a single number measuring
    how error of fitted distribution compared to target (original data).

    Flat because scipy's L-BFGS-B optimiser only takes flat arrays
    -> [centre1, height1, width1, centre2, height2, width2, for all peaks]
    """
    #Unpack the flat array back into a list of (centre, height, width) tuples.
    peak_list = []
    for peak_index in range(number_of_peaks):
        start = peak_index * 3        #each peak takes 3 slots, use start as point to find groups of 3 in the flat list
        center_hour = flat_parameters[start + 0]
        height = flat_parameters[start + 1]
        width = flat_parameters[start + 2]
        peak_list.append((center_hour, height, width)) #add the full tuple to the peak list

    fitted_distribution = multi_peak_distribution(peak_list) #fit the list with the mpd function

    #Use sum of squared errors to measure fit of the fitted distributions
    squared_errors = (fitted_distribution - target_probabilities) ** 2
    total_error = squared_errors.sum()
    return total_error


#------------------------------------------------- 
# CHOOSE INITIAL STARTING POINT FOR THE OPTIMIZER
#-------------------------------------------------
def make_initial_guess(target_probabilities, number_of_peaks):
    """
    Function to make an educated guess on the first estimation (x0 paramerter in L-BFGS-B). First of three attempts
    at finding a starting point
    Find the hours with the highest probabilities and place one Gaussian
    on each of those hours to start from.
    """
    #Find number_of_peaks hours with the highest probability.
    sorted_hours_by_prob = np.argsort(target_probabilities)[::-1]   #argsort goes from low to high, so flip the sequence
    top_hours = sorted_hours_by_prob[:number_of_peaks] #get few highest peaks, determined by amount of peaks under investigation

    initial_parameters = []
    for hour in top_hours:
        initial_center = float(hour) #will turn into float later on, changing preventively just in case
        initial_height = float(target_probabilities[hour]) * number_of_peaks * 3 #rescale to prevent the optimizer starting from a near-flat curve, where 3 is hyperparameter
        initial_width  = 1.5          #just an initial value for now
        initial_parameters.extend([initial_center, initial_height, initial_width]) #add the peak to the init guess

    return np.array(initial_parameters)


#-------------------  
#FITTING N PEAKS
#------------------- 

def fit_n_peaks(target_probabilities, number_of_peaks):
    """
    Uses scipy's L-BFGS-B optimiser to find the best (center, height, width)
    for n-peak Gaussians.

    L-BFGS-B is a gradient-based method which works with the 24h bounds.

    This is ran multiple times from different starting points to reduce the
    risk of getting stuck in a local minimum.
    
    First time: Uses make_initial_guess, which places Gaussians on the hours with the highest raw probabilities.
    Second time: Uses find_peaks, which looks for local maxima rather than just the globally tallest hours. It considers the neighborhood.
    Third time: Random trials scatter starting points across the entire parameter space.

    Returns the best peak list (tuple with (time, peak, spread)s, for as many peaks as allowed) and error of those peaks
    """
    #Bounds for each peak parameter
    #center: anywhere on the 24-hour clock (where 23 = 23:00-00:00 again)
    #height: magnitude capped at 8 
    #width : at least 0.3 (very sharp spike) up to 5 hours (very broad)
    #These bounds were iteratively tested and selected through visually comparing the final estimation with actual probabilies with a graph
    
    bounds_per_peak = [(0.0, 23.0), (0.05, 8.0), (0.3, 5.0)]
    all_bounds = bounds_per_peak * number_of_peaks #intialize n bounds for the three values in the n peaks
    
    best_error_so_far = np.inf
    best_flat_params  = None
    
    #FIRST TEST (Educated Guess)
    #Set starting point x0, with the init guess function
    initial_guess_topN = make_initial_guess(target_probabilities, number_of_peaks)

    #Initialize and use the model with minimize from the scipy.optimize package
    #For documentation refer to both:
    # - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    # - https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
    
    result = minimize(
        fun = sum_of_squared_errors,
        x0 = initial_guess_topN,
        args = (number_of_peaks, target_probabilities),
        method = "L-BFGS-B",
        bounds = all_bounds,
        options = {"maxiter": 2000},
    )
    
    if result.fun < best_error_so_far: #check if it reached a lower sum of squared than current best, if yes, change standings
        best_error_so_far = result.fun 
        best_flat_params  = result.x

    #SECOND TEST (Use peaks)
    #Find_peaks looks for local maxima in the probability array.
    #For documentation refer to:
    # - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
    #returns indexes of peaks + properties of said peak
    
    detected_peak_indices, _ = find_peaks(target_probabilities, prominence = 0.0)
    #prominence required to prevent IndexError

    if len(detected_peak_indices) >= number_of_peaks:
        #If enough peaks were detected, pick the tallest ones
        peak_heights = target_probabilities[detected_peak_indices]
        tallest_first = np.argsort(peak_heights)[::-1]
        chosen_peak_indices = detected_peak_indices[tallest_first[:number_of_peaks]]
    else:
        #Fall back to the top-probability hours if detection found too few
        chosen_peak_indices = np.argsort(target_probabilities)[::-1][:number_of_peaks]

    initial_guess = []
    for index in chosen_peak_indices:
        initial_guess.extend([float(index),
                                       float(target_probabilities[index]) * number_of_peaks * 3,
                                       1.5]) #pass the (t, h, w) tuple to the initial guess 
    initial_guess = np.array(initial_guess)
    #same procedure as for first round, now with another initial guaas
    result = minimize(
        fun = sum_of_squared_errors,
        x0 = initial_guess,
        args = (number_of_peaks, target_probabilities),
        method = "L-BFGS-B",
        bounds = all_bounds,
        options = {"maxiter": 2000},
    )
    if result.fun < best_error_so_far:
        best_error_so_far = result.fun
        best_flat_params  = result.x

    #THIRD TEST (Random Peaks)
    #Random restarts could escape local minima that both structured guesses above may strand in
    rng = np.random.default_rng(seed=42) #For reproducability
    for a in range(10):  #10 is a hyperparameter for the amount of random trials
        random_guess = []
        for b in range(number_of_peaks): #set random peaks, stay within the bounds I defined earlier but dont have height max out at extreme values, can move in minimizer if it does occur
            random_centre = rng.uniform(0, 23)
            random_height = rng.uniform(0.1, 5.0)
            random_width = rng.uniform(0.5, 3.0)
            random_guess.extend([random_centre, random_height, random_width])
        random_guess = np.array(random_guess)
    #Run the random guess through the minimizer
        result = minimize(
            fun = sum_of_squared_errors,
            x0 = random_guess,
            args = (number_of_peaks, target_probabilities),
            method = "L-BFGS-B",
            bounds = all_bounds,
            options = {"maxiter": 2000},
        )
        if result.fun < best_error_so_far:
            best_error_so_far = result.fun
            best_flat_params  = result.x

    #Unpack the best flat parameter array into a readable list of tuples, just as in the objective function
    best_peak_list = []
    for peak_index in range(number_of_peaks):
        offset = peak_index * 3
        centre = best_flat_params[offset + 0]
        height = best_flat_params[offset + 1]
        width  = best_flat_params[offset + 2]
        best_peak_list.append((centre, height, width))

    return best_peak_list, best_error_so_far
    
def select_best_npeaks(target_probabilities, max_peaks=4):
    """
    Tries fitting up to n peaks.
    The Bayesian Information Criterion (BIC) gives a way to trade
    off fit quality against model complexity. (Schwarz, 1978, pp. 461–464)

    BIC = N * ln(RSS / N)  +  k * ln(N)
    
    where:
    RSS = sum of squared residuals (lower = better fit)
    N = number of data points (24 hours)
    k = number of free parameters (3 per peak: centre, height, width)
 
    The first term rewards a good fit.  
    The second term penalizes complexity: adding one more peak adds 3 parameters, so the BIC goes up by 3*ln(24) = 9.6
     
    Returns the peak list that minimises the BIC.
    
    """
    N = len(target_probabilities)

    best_bic = np.inf
    best_peak_list = None
    best_n = None

    for n_peaks in range(1, max_peaks + 1):
        peak_list, RSS = fit_n_peaks(target_probabilities, n_peaks)

        k = n_peaks * 3 #center + height + width per peak
        bic = (N * np.log(RSS / N) + k * np.log(N)) #plug them into formula

        if bic < best_bic:
            best_bic = bic
            best_peak_list = peak_list
            best_n = n_peaks

    return best_peak_list, best_n, best_bic


#---------------------------------------
# RUN THE FITTING FOR EVERY APPLIANCE
#---------------------------------------

print("Fitting Gaussian peaks to each appliance...\n")

fitted_peaks_per_appliance = {}

for appliance_name, target_probabilities in normalized_data.items():
    peak_list, n_peaks_chosen, bic_score = select_best_npeaks(target_probabilities)
    fitted_peaks_per_appliance[appliance_name] = peak_list

    fitted_dist = multi_peak_distribution(peak_list) #convert to valid distribution
    rss = np.sum((fitted_dist - target_probabilities)**2)   #final rss
    print(f"  {appliance_name:<16}  {n_peaks_chosen} peaks   RSS = {rss:.6f}   BIC = {bic_score:.3f}")

print("\n")


#-------------------------------------------
# PRINT READY-TO-USE VARIABLE DEFINITIONS
#-------------------------------------------

#Map the names to variable names in the copy-paste text.
variable_names = {
    "Dishwasher": "dishwasher_raw_peaks",
    "Washing": "washing_raw_peaks",
    "Tumble Drier": "tumble_drier_raw_peaks",
    "Cooker": "cooker_raw_peaks",
    "Oven": "oven_raw_peaks",
    "Grill": "grill_raw_peaks",
    "Hob": "hob_raw_peaks",
    "TV Total": "tv_raw_peaks",
    "Electronics": "electronics_raw_peaks"
}

print("=" * 60)
print("Copy-paste below values into final simulation:\n")

for appliance_name, peak_list in fitted_peaks_per_appliance.items():
    variable_name = variable_names[appliance_name]
    print(f"{variable_name} = ({peak_list})\n")

ev_peaks = [               #manually added and visually approximated Robinson et al. (2013) figure 6: blue line "Home Private"
    (0.6, 2.2, 1.8),   
    (14.0, 0.5, 3),   
    (19.5, 2.5, 2.3),
    (24, 1, 2.7),
]

print(f"ev_raw_peaks = ({ev_peaks})")

#-------------------------
#PLOTTING
#-------------------------
hours = np.arange(24, dtype=float)

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle("Original probabilities vs. Gaussian fit", fontsize=13)

# Flatten the 3x3 grid of axes into a single list for easy iteration.
axes_flat = axes.flatten()

for axis, (appliance_name, target_probabilities) in zip(axes_flat, normalized_data.items()):
    peak_list = fitted_peaks_per_appliance[appliance_name]
    fitted_dist = multi_peak_distribution(peak_list)
    residual_ss = np.sum((fitted_dist - target_probabilities)**2)
    number_of_peaks = len(peak_list)

    axis.bar(hours, target_probabilities, alpha=0.4, color="steelblue", label="Original data")
    axis.plot(hours, fitted_dist, color="red", linewidth=2,
              label=f"Fit ({number_of_peaks} peaks)")

    # Mark each peak centre with a vertical dashed line.
    for centre_hour, height, width in peak_list:
        axis.axvline(x=centre_hour, color="red", alpha=0.25, linewidth=1, linestyle="--")

    axis.set_title(f"{appliance_name}  (RSS={residual_ss:.5f})", fontsize=10)
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("Probability")
    axis.set_xticks(range(0, 24, 3))
    axis.legend(fontsize=8)

plt.tight_layout()
