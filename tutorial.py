import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

N = 1000 # population being modeled
I0 = 1 # intial infected
R0 = 0 # intial recovered
E0 = 0 # intial exposed
S0 = N - I0 - R0 - E0 # intial suceptible

beta = 0.3 # contact rate
sigma = 1/5.2 # incubation rate
gamma = 1/10 # mean recovery rate

# time points (days) over which to predict
t = np.linspace(0,160, 160) # (start, end, points)

# SEIR differential equations
def deriv(t, y, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

y0 = S0, E0, I0, R0 # intial conditions

sol = solve_ivp(deriv, [0,160], y0, args=(N, beta, sigma, gamma), t_eval=t)

S, E, I, R = sol.y # extract results

# plotting
plt.figure(figsize=(20,11))
plt.plot(t,S,'b',label="Susceptible")
plt.plot(t, E, "y", label = "Exposed")
plt.plot(t, I, "r", label = "Infected")
plt.plot(t, R, "g", label="Recovered")
plt.title("SEIR Model")
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid(lw=2,ls=":")
plt.show() 