import numpy as np

N = 10
A, B, C = 500, 500, 1
max_iterations = 10000
restarts = 10
seed = 42
np.random.seed(seed)
distances = np.random.randint(1, 100, (N, N))
np.fill_diagonal(distances, 0)

def energy(state):
    E1 = A * np.sum((np.sum(state, axis=1) - 1) ** 2)
    E2 = B * np.sum((np.sum(state, axis=0) - 1) ** 2)
    E3 = 0
    for i in range(N):
        for p in range(N):
            next_pos = (p + 1) % N
            for k in range(N):
                E3 += C * distances[i, k] * state[i, p] * state[k, next_pos]
    return E1 + E2 + E3

def update_one(state, i, p):
    row_sum = np.sum(state[i, :]) - state[i, p]
    col_sum = np.sum(state[:, p]) - state[i, p]
    succ = np.sum(distances[i, :] * state[:, (p + 1) % N])
    pred = np.sum(distances[:, i] * state[:, (p - 1) % N])
    h = 2 * A * row_sum + 2 * B * col_sum + C * (succ + pred) - (A + B)
    return 1 if h < 0 else 0

best_state = None
best_energy = float('inf')
for r in range(restarts):
    state = np.random.randint(0, 2, (N, N))
    for iteration in range(max_iterations):
        changed = False
        indices = [(i, p) for i in range(N) for p in range(N)]
        np.random.shuffle(indices)
        for i, p in indices:
            new_val = update_one(state, i, p)
            if new_val != state[i, p]:
                state[i, p] = new_val
                changed = True
        if not changed:
            break
    E = energy(state)
    if E < best_energy:
        best_energy = E
        best_state = state.copy()

def validate_tour(state):
    row_sums = np.sum(state, axis=1)
    col_sums = np.sum(state, axis=0)
    return np.all(row_sums == 1) and np.all(col_sums == 1)

valid = validate_tour(best_state)
print("Distance Matrix:")
print(distances)
print("\nFinal State (Best Found):")
print(best_state)
if valid:
    print("\nValid Tour Found!")
else:
    print("\nInvalid Tour.")
print("\nNeurons:", N * N)
print("Unique symmetric weights:", (N * N) * (N * N - 1) // 2)
print("Thresholds:", N * N)
