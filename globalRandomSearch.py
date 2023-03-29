from random import uniform

def global_random_search(f, n, bounds, N=1000, is_timed=False):
    l = [r[0] for r in bounds]
    u = [r[1] for r in bounds]

    best_params = None
    min_cost = float("inf")

    xs, fs = [], []

    for _ in range(N):
        sample = [uniform(l[i], u[i]) for i in range(n)]
        cost = f(*sample)
        if cost < min_cost:
            min_cost = cost
            best_params = sample[:]

        if not is_timed:
            xs.append(best_params[:])
            fs.append(min_cost)

    return xs, fs
