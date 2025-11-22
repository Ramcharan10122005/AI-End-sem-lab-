import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import yfinance as yf
import os

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Download stock data
ticker = "AAPL"
data = yf.download(ticker, period="10y")

# FIX: Use 'Close' because 'Adj Close' no longer exists
data['Return'] = np.log(data['Close'] / data['Close'].shift(1))
data = data.dropna()

# Prepare return values
returns = data['Return'].values.reshape(-1, 1)

# Fit HMM with 2 states
model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
model.fit(returns)

# Predict hidden states
hidden_states = model.predict(returns)

# Save transition matrix printout
print("\n=== Transition Matrix ===")
print(model.transmat_)

# Plot hidden states over time
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'])

for i in range(model.n_components):
    state_idx = (hidden_states == i)
    plt.scatter(data.index[state_idx], data['Close'][state_idx])

plt.title("Hidden Market States")
plt.savefig("outputs/hidden_states.png")
plt.close()

# Plot returns colored by state
plt.figure(figsize=(12,6))
for i in range(model.n_components):
    state_idx = (hidden_states == i)
    plt.scatter(data.index[state_idx], data['Return'][state_idx], s=5)

plt.title("Returns Colored by Hidden State")
plt.savefig("outputs/returns_states.png")
plt.close()

print("\n=== DONE! All plots saved to './outputs/' ===")
