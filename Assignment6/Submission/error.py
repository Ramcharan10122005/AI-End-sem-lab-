import numpy as np

def build_rook_weights(sz=8, A=100):
    n = sz * sz
    wt = np.zeros((n, n))
    theta = np.zeros(n)
    for i in range(sz):
        for p in range(sz):
            a = i * sz + p
            for q in range(sz):
                if p != q:
                    b = i * sz + q
                    wt[a, b] += 2 * A
            for k in range(sz):
                if k != i:
                    b = k * sz + p
                    wt[a, b] += 2 * A
            theta[a] = -(A + A)
    return wt, theta

def hopfield_async_run(state, wt, theta, max_updates=10000, seed=42):
    np.random.seed(seed)
    n = state.size
    s = state.flatten().copy()
    for _ in range(max_updates):
        idx = np.random.randint(0, n)
        h = np.dot(wt[idx], s) + theta[idx]
        new = 1 if h < 0 else 0
        if new != s[idx]:
            s[idx] = new
        else:
            # check if no change will occur after a full sweep
            pass
    return s.reshape(state.shape)

def is_valid_rook(board):
    rows = np.sum(board, axis=1)
    cols = np.sum(board, axis=0)
    return np.all(rows == 1) and np.all(cols == 1)

def experiment_rook_error(sz=8, A=100, trials=200, max_k=20, seed=42):
    np.random.seed(seed)
    wt, theta = build_rook_weights(sz, A)
    base = np.zeros((sz, sz), dtype=int)
    perm = np.random.permutation(sz)
    for i in range(sz):
        base[i, perm[i]] = 1
    print("Base placement:")
    print(base)
    results = []
    n = sz * sz
    for k in range(1, max_k + 1):
        exact_count = 0
        valid_count = 0
        for t in range(trials):
            s = base.flatten().copy()
            flip_pos = np.random.choice(n, k, replace=False)
            s[flip_pos] = 1 - s[flip_pos]
            s = s.reshape((sz, sz))
            final = hopfield_async_run(s, wt, theta, max_updates=2000, seed=seed + t)
            if np.array_equal(final, base):
                exact_count += 1
            if is_valid_rook(final):
                valid_count += 1
        results.append((k, exact_count / trials, valid_count / trials))
    print("k, exact_recovery_rate, any_valid_recovery_rate")
    for r in results:
        print(r[0], r[1], r[2])

np.random.seed(42)
experiment_rook_error()
