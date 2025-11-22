import numpy as np

def rook_net(sz=8, n_it=10000, A=100, n_r=8, seed=42):
    np.random.seed(seed)
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
    state = np.zeros(n, dtype=int)
    rpos = np.random.choice(n, n_r, replace=False)
    for p in rpos:
        state[p] = 1
    print("Initial State:")
    print(state.reshape(sz, sz))
    for it in range(n_it):
        changed = False
        idx = np.arange(n)
        np.random.shuffle(idx)
        for a in idx:
            h = np.dot(wt[a], state) + theta[a]
            new = 1 if h < 0 else 0
            if new != state[a]:
                state[a] = new
                changed = True
        if not changed:
            break
    print("Final State:")
    print(state.reshape(sz, sz))
    return state.reshape(sz, sz)

np.random.seed(42)
sol = rook_net()
