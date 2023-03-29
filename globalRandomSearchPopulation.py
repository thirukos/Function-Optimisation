from random import uniform

def global_random_search_population(f, n, bounds, N=1000, M=10, num_iters=10, neighborhood_ratio=0.1, is_timed=False):
    # Extract lower and upper bounds for each parameter
    l = [r[0] for r in bounds]
    u = [r[1] for r in bounds]

    # Initialize best parameters list and minimum cost list
    best_params_list = [[uniform(l[i], u[i]) for i in range(n)] for _ in range(M)]
    min_cost_list = [float("inf")] * M

    xs, fs = [], []

    # Generate N random parameter sets and calculate their costs
    for _ in range(N):
        sample = [uniform(l[i], u[i]) for i in range(n)]
        cost = f(*sample)

        # Check if the current sample is better than any of the stored best samples, and update the best_params_list if necessary
        for i in range(M):
            if cost < min_cost_list[i]:
                min_cost_list.insert(i, cost)
                best_params_list.insert(i, sample[:])
                min_cost_list.pop()
                best_params_list.pop()
                break

        # Store the best parameters and costs for tracking (if is_timed is False)
        if not is_timed:
            xs.append(best_params_list[:])
            fs.append(min_cost_list[:])

    # Perform the population-based sampling iterations
    for _ in range(num_iters):
        # Initialize new lists to store the best parameter sets and their corresponding costs for this iteration
        new_best_params_list = [[uniform(l[i], u[i]) for i in range(n)] for _ in range(M)]
        new_min_cost_list = [float("inf")] * M

        # Iterate over each of the current M best points
        for j in range(M):
            # Generate N random parameter sets around the current best point (in its neighborhood)
            for i in range(N):
                sample = [uniform(l[i] + neighborhood_ratio * (best_params_list[j][i] - l[i]),
                  u[i] - neighborhood_ratio * (u[i] - best_params_list[j][i])) for i in range(n)]
                cost = f(*sample)
                # Check if the current sample is better than any of the stored best samples for this iteration, 
                # and update the new_best_params_list if necessary
                for i in range(M):
                    if cost < new_min_cost_list[i]:
                        new_min_cost_list.insert(i, cost)
                        new_best_params_list.insert(i, sample[:])
                        new_min_cost_list.pop()
                        new_best_params_list.pop()
                        break

        # Update the best_params_list and min_cost_list for the next iteration
        best_params_list = new_best_params_list[:]
        min_cost_list = new_min_cost_list[:]

        if not is_timed:
            xs.append(best_params_list[:])
            fs.append(min_cost_list[:])

    return xs, fs
