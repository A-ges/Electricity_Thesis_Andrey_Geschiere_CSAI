import numpy as np

def generate_daily_contacts(full_network, mean_contacts = 8.0, std_contacts = 3, day_seed = 0):
    """
    Derive a daily contact sub-network from each agent's full social network.

    Mean somewhat derived from Mousa et al. (2021)
    -> The study analyzed daily contacts per person across different circumstances (Home, School, Work, Other) per age group
        -> Their mean landed around 12 contacts per person per day
        CONSIDERING:
            -> My study views househoulds as a singular agent, not considering inhabitants
            -> Talking about electricity usage is a uncommon conversation topic
        DECISION:
            -> mean set at 8 and std at 3 produces an actual mean close to 10 daily contacts influencing electricity usage per household (as seen in bottom check) with some heterogeneity
            -> this mean increases as N grows (because networks less bounding) but stabilizes at higher N's (from 250 onwards)
    
    Each day every agent talks to a subset of their full network
    This algorithm guarantees:
      - Every agent has at least 1 daily contact
      - Selection is mutual: if X picks Y, Y automatically picks X too
      - It still works for a small N (But with a network of 50, there are very little agents who have 10 connections in total, automatically leading to lower means)
      
    Parameters
    -> full_network: dictionary containing agent_id: [list of neighbour agent ids], given by the file named networks.json
    -> mean_contacts: Target mean number of daily contacts per agent 
    -> std_contacts: Standard deviation of daily contact count
    -> day_seed: random state for this day

    Returns a dictionary like the input, where agent_id = [list of daily contact agent ids]
    """
    rng = np.random.default_rng(day_seed)
    agents = list(full_network.keys()) #get all individual agents from network.json

    daily = {} #used to store day networks for all agents 
    for agent in agents:
        daily[agent] = set()  #key is agent, value is for now an empty set, to be added with agents

    #Shuffle agents, so no positional bias in who gets to pick first   
    #Agents processed later may already have contacts assigned to them by earlier agents (mutual picks), so they have fewer new picks
    
    shuffled_agents = rng.permutation(agents) #refer to https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html

    for agent in shuffled_agents:
        neighbours = full_network[agent]  #assign all contacts as neigbors first

        #sample a personal target, capped to the agent's actual degree
        raw_target = rng.normal(mean_contacts, std_contacts) #get the amount of people this agent would ideally have, conform the parameters
        target = int(np.clip(round(raw_target), 1, len(neighbours))) #talk to at least one person and make it a whole int

        already_have = len(daily[agent]) #filled by mutual picks
        if already_have >= target:      #if mutual picks exceed chosen target, satisfied, next agent and this one is done
            continue

        needed = target - already_have #how much left to fill to get to target

        #Only consider neighbours not already in today's contact list
        available = [n for n in neighbours if n not in daily[agent]]
        if not available: #if not agents left to pick from, dont bother
            continue

        n_pick = min(needed, len(available)) #determine final picks (when considering perhaps all agents might have been used up)
        chosen = rng.choice(available, size=n_pick, replace=False)

        for contact in chosen:
            daily[agent].add(contact)   #add contact to self
            daily[contact].add(agent)   #mutual assignment

    #Extra pass to make sure all agents have at least one contact
    for agent in agents:
        if len(daily[agent]) == 0:
            neighbours = full_network[agent]
            if neighbours:
                picked = rng.choice(neighbours)
                daily[agent].add(picked)
                daily[picked].add(agent)

    result = {}
    for agent, contacts in daily.items():
        result[agent] = list(contacts)
    return result

"""
#Uncomment for test
import json

with open("networks.json", "r") as f:
    data = json.load(f)
    network = data["250d"]

net = generate_daily_contacts(network, day_seed = 80)
total = 0
for agent in net:
    total += len(net[agent])

print(f"Mean = {total/len(net)}")
print(net["AG003"])
print(net["AG039"])
"""
