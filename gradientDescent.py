def gradient_descent(f, df, n, x_init, alpha=0.01, N=1000, is_timed=False):
    x_current = x_init.copy()
    func_current = f(*x_current)
    xs, fs = [], []

    if not is_timed:
        xs.append(x_current.copy())
        fs.append(func_current)
        
    for _ in range(N + 1):

        if not is_timed:
            xs.append(x_current.copy())
            fs.append(func_current)

        for i in range(n):
            x_current[i] -= alpha * df[i](x_current[i])
        func_current = f(*x_current)

    
    return xs, fs