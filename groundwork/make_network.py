import random
import math

"""
This code contains the make network function
It generates a heterogeneous clustered network for using the approach from House (2014)
The Claude LLM has been used to make sense of the paper and lay down this approach through pseudocode
"""

def make_network(agents, link_density=0.08, degree_heterogeneity=1.5, clustering_coef=0.25, random_state=42):
    """
    Generate a heterogeneous clustered random network for a list of agents, based on the method from House (2014).
    The network is created with a Markov Chain Monte Carlo (MCMC) algorithm:
    -> Approach every decision based on current states only
    -> Randomly accept or decline edges (required for feasibility and still provides very good outcomes according to House (2014).
    -> Until model converges around the set parameters
        -> Determined by the Hamiltonian function
 
    Parameters:
    - agents: List of agent IDs in the simulation

    - link_density: float 
        -> Target fraction of all POSSIBLE connections that actually exist 
            -> With 150 agents and density 0.08 (150 x 0.08) = 12 neigbors per agent on average
                -> Scales along with the N 

    - degree_heterogeneity: float
        -> degree_heterogeneity = Var(k) / mean(k)
            -> The variance-to-mean ratio of the degree distribution. 
            -> A value of 0 means every agent has the same number of connections
            -> A value of 1 means mean == variance, no real structure, random
            -> 1+ = wider spread, gives more real social hubs and some with lower social circles

    - clustering_coef: float
        -> Target clustering coefficient 
            -> Probability that two of an agent's neighbors are also connected to each other -> "friends of friends are friends". 
        
    - random_state: set for reproducability

    Returns a dictionary where each key is the agent ID and each value is a list of neighbor IDs. 
    If X appears in Y's list, Y will also appear in X's list
    """
    random.seed(random_state)
    N = len(agents)
    
    
    index_to_agent = {}     
    for index, agent in enumerate(agents):
        index_to_agent[index] = agent

    G = []                #G will function as the adjacency matrix, an agents x agents binary map, where 1 corresponds to a connection (G[i][j] == 1)     
    for i in range(N):    #Mappings to self will turn out to be 0's, for now just initialize
        G.append([0] * N) #for every agent, add a row of N zero's 

    #-------------------------------------
    #Computing Hamiltonian beta values 
    #Used to plug into the Hamiltonian H(G) = beta_m*(...)^2 + beta_l*(...)^2 + beta_t*(...)^2
    #"Higher values of these parameters will attribute lower probabilities to networks further from the mode" (House, 2014)

    #Values/formulas are from House (2014), equation 12:
    #   beta_m = 1 / (2*sigma^2) 
    #   beta_l = (theta_l + theta_m*(N-1) − 1) * beta_m
    #   beta_t = theta_t * beta_l
    #------------------------------------

    beta_m = 0.25 #0.25 was found to perform well according to House (2014)
    beta_l = (degree_heterogeneity + link_density * (N - 1) - 1) * beta_m
    beta_t = clustering_coef * beta_l

    #--------------------------------------------------------------------
    #Defining the Hamiltonian, refer to equation 11 from House 2014
    #
    #H(M, L, T) measures how far the current network is from the three
    #parameter set targets. Lower H = network is closer to parameters.
    #--------------------------------------------------------------------

    def hamiltonian(M, L, T):
        """
        Compute Hamiltonian given the current network statistics

        Parameters:
        - M: int, total directed links -> Where A is with B
        - L: int, total directed lines -> Where A is with B and B with C
        - T: int, total directed triangles -> where A is with B and C, B is with A and C, C is with A and B.

        Returns Hamiltonian value. Lower H = network is closer to parameters.
        """

        term_m = beta_m * (M - (link_density * N * (N - 1))) ** 2

        term_l = beta_l * (L - (M * ((degree_heterogeneity - 1) + M / N))) ** 2

        term_t = beta_t * (T - (clustering_coef * L)) ** 2

        return term_m + term_l + term_t

    #----------------------------------------------------------------------------
    #Updating of M, L, T
    #
    #Recomputing MLT from scratch at every MCMC step would be O(N^2) per
    #step. Instead, compute only the change caused by flipping edge (i,j)
    #----------------------------------------------------------------------------

    def delta_statistics(i, j):
        """
        Compute the change in (M, L, T) caused by flipping edge (i, j)

        Returns a tuple: (delta_M, delta_L, delta_T)
        """

        flip = 1 - 2 * G[i][j] #turns 0 into 1 and 1 into 0, based on current adjacency matrix status

        #Each undirected edge flip changes directed M by 2
        delta_M = 2 * flip

        ki = sum(G[i])     #current degree of node i, before flip
        kj = sum(G[j])     #current degree of node j, before flip
        
        delta_L = 2 * flip * (ki - G[i][j] + kj - G[j][i])

        #A new undirected triangle is formed for every node k that is currently connected to both i and j

        common_neighbors = 0
        for k in range(N):
            if k != i and k != j:
                #k is a common neighbor if edges (i,k) and (j,k) both exist

                if G[i][k] == 1 and G[j][k] == 1:
                    common_neighbors += 1

        delta_T = flip * 6 * common_neighbors #times 6 because of undirected (3 edges times 2 ways)

        return (delta_M, delta_L, delta_T)
    
    steps = max((N**2) * 20, 20000)   #at least 20k steps, 20 is a hyper parameter. Higher values take longer to process and the gain in approaching the target turns out very small


    #initial Edge (M), Line (L) Triangle (T) count on the empty adjacency matrix + the initial hamiltonian
    current_M = 0   
    current_L = 0    
    current_T = 0    
    current_H = hamiltonian(current_M, current_L, current_T)

    all_edges = [] #List containing all possible edges, used to randomly draw from rather than iterating over full search space every single time
    for i in range(N):
        for j in range(i + 1, N):
            all_edges.append((i, j)) 
    
    #----------------------------------------------------------
    #Loop with the MCMC algorithm -> Refer to Equation 13 from House (2014)
    #
    #At each step:
    #   1. Pick a random edge (i, j) to consider flipping
    #   2. Compute the change in Hamiltonian ΔH = H_proposed − H_current
    #   3. Always accept the flip if delta_Hamiltonian <= 0 (proposed state is closer to targets)
    #   -> IF proposed state is worse -> Accept the flip with probability exp(-delta_H) 
    #----------------------------------------------------------

    for step in range(steps):

        #Pick a random edge to consider flipping
        i, j = random.choice(all_edges)

        #Compute how MLT would change if this edge is flipped
        delta_M, delta_L, delta_T = delta_statistics(i, j)

        #Compute proposed new statistics
        new_M = current_M + delta_M
        new_L = current_L + delta_L
        new_T = current_T + delta_T

        #Compute how the Hamiltonian would change
        new_H = hamiltonian(new_M, new_L, new_T)
        delta_H = new_H - current_H

        if delta_H <= 0:
            #Moving to a better state = always accept
            accept = True
        else:
            #Moving to a worse state = accept with probability exp(-delta_H)
            accept = random.random() < math.exp(-delta_H)

        #Apply the accepted flip to the adjacency matrix
        if accept:
            new_val = 1 - G[i][j] 
            G[i][j] = new_val         
            G[j][i] = new_val       

            current_M = new_M
            current_L = new_L
            current_T = new_T
            current_H = new_H

    #-------------------------------------------
    # Converting to dictionary format
    #-------------------------------------------

    network = {}
    for index in range(N):
        agent = index_to_agent[index]
        neighbors = []
        for j in range(N):
            if G[index][j] == 1:
                neighbors.append(index_to_agent[j])
        network[agent] = neighbors

    for agent in agents:                      #Addition that causes no sitations of an empty network, which could occur with smaller N's
        if len(network[agent]) == 0:          #Mostly triggered when using small sizes
            filler = random.choice(agents)
            while filler == agent:
                filler = random.choice(agents)
            network[agent].append(filler)
            network[filler].append(agent)
    return network

import collections

def copyable_networks(agents):
    
if __name__ == "__main__":

    

    #Build a population
    test_agents = ["AG" + str(i).zfill(3) for i in range(agents)]

    print("Building network....")
    net = make_network(test_agents, link_density=0.08, degree_heterogeneity=1.5, clustering_coef=0.25, random_state=80)

    # --- Basic stats ---
    degrees = [len(neighbors) for neighbors in net.values()]
    total_undirected_edges = sum(degrees) // 2
    N = len(test_agents)
    actual_density = total_undirected_edges / (N * (N - 1) / 2)

    print(f"\n--- Network summary ---")
    print(f"Agents          : {N}")
    print(f"Total edges     : {total_undirected_edges}")
    print(f"Actual density  : {actual_density:.4f}  (target: {0.8}")
    print(f"Mean degree     : {sum(degrees)/N:.2f}")
    print(f"Max degree      : {max(degrees)}")
    print(f"Min degree      : {min(degrees)}")
    print(f"Degree std dev  : {(sum((d - sum(degrees)/N)**2 for d in degrees)/N)**0.5:.2f}")

    # Degree heterogeneity (variance/mean)
    mean_k = sum(degrees) / N
    var_k  = sum((d - mean_k)**2 for d in degrees) / N
    actual_heterogeneity = var_k / mean_k if mean_k > 0 else 0
    print(f"Actual hetero.  : {actual_heterogeneity:.3f}  (target: 1.5)")

    # Clustering coefficient (fraction of closed triangles over open ones)
    def local_cc(node_index, G, N):
        nbrs = []
        for j in range(N):
            if G[node_index][j] == 1:
                nbrs.append(j)
        k = len(nbrs)
        if k < 2:
            return 0.0
        links = 0
        for a in range(k):
            for b in range(a + 1, k):
                links += G[nbrs[a]][nbrs[b]]
        return 2 * links / (k * (k - 1))

    G_mat = []
    for _ in range(N):
        G_mat.append([0] * N)

    index_map = {}
    for i, a in enumerate(test_agents):
        index_map[a] = i

    for agent, nbrs in net.items():
        i = index_map[agent]
        for nb in nbrs:
            j = index_map[nb]
            G_mat[i][j] = 1

    cc_values = []
    for i in range(N):
        cc_values.append(local_cc(i, G_mat, N))

    valid_cc = []
    for v in cc_values:
        if v > 0:
            valid_cc.append(v)

    mean_cc = sum(valid_cc) / len(valid_cc) if valid_cc else 0
    print(f"Actual clust.   : {mean_cc:.3f}  (target: 0.25)")

    for agent in test_agents:
        print(agent, net[agent])
