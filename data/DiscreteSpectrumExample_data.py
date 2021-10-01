"""
Create training/validation data for discrete spectrum example.

All data comes from solutions to a simple ODE.
Training data:
    Initial conditions:
        3,500 ICs randomly selected in the square:
        [-.5, .5] x [-.5, .5]
    Solve from t = 0 to 1 in steps of 0.02
    Parameters mu = -0.05, lambda = -1
Validation data:
    Same structure as training data but with 1,000 ICs
"""

import numpy as np
from scipy.integrate import solve_ivp
from numpy.random import default_rng

def func(t, x):
    mu = -0.05
    lam = -1
    return [mu*x[0], lam*(x[1] - x[0]**2)]

def create_data(numICs, seed):
    rng = default_rng(seed)
    
    # select a bunch of initial conditions in the correct range
    ic_ranges = np.array([[-.5, .5], [-.5, .5]])
    ic_spans = ic_ranges[:,1] - ic_ranges[:,0]
    
    initial_conds = rng.random((numICs,2))
    initial_conds = initial_conds * ic_spans + ic_ranges[:,0]
    
    lenT = 51
    data = np.zeros((numICs, lenT, 2))
    
    t_span = [0, 1]
    t = np.linspace(0, 1, lenT) # tSpan = 0:0.02:1
    for j in np.arange(numICs):
        sol = solve_ivp(func, t_span, initial_conds[j,:], t_eval=t)
        data[j,:,:] = (sol.y).transpose() # shape [2, 51]
    return data

data_prefix = 'DiscreteSpectrumExample'
n_IC_train = 3500  # Number of initial conditions
n_IC_val = 1000

train_data = create_data(n_IC_train, seed=1)
np.save("DiscreteSpectrumExample_train1_x.npy", train_data, allow_pickle=False)

val_data = create_data(n_IC_val, seed=2)
np.save("DiscreteSpectrumExample_val_x.npy", val_data, allow_pickle=False)

