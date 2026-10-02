import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import yaml


filepath = "ConfigFiles/config.yaml"
with open(filepath, "r") as file:
    data = yaml.safe_load(file)

N = data["population"]["N"]
M0 = data["initial"]["M0"] # maternal immune infants 
E0 = data["initial"]["E0"] # exposed in non-infectious latent period
I0 = data["initial"]["I0"] # infectious individuals
R0 = data["initial"]["R0"] # recovered with temporary immunity -> suceptible
S0 = N - M0 - E0 - I0 - R0 # suceptible


lam = data["parameters"]["lam"] # recruitment rate (m births)
delta = data["parameters"]["delta"] # maternal immunity rate
beta = data["parameters"]["beta"] # transmission coefficient (contact rate)
sigma = data["parameters"]["sigma"] # contact number (incubation rate)
gamma = data["parameters"]["gamma"] # recovery rate (1/gamma - avg infectious period)
omega = data["parameters"]["omega"] # immunity rate (R -> S)
mu = data["parameters"]["mu"] # natural per capita mortality rate

t = np.linspace(data["simulation"]["t_start"], data["simulation"]["t_end"], data["simulation"]["t_points"]) # start, stop, points

def deriv (t, y, N, lam, delta, beta, sigma, gamma, omega, mu):
    M, S, E, I, R = y
    dMdt = lam - (delta * M) - (mu * M)
    dEdt = (beta * I * S)/N - ((sigma + mu) * E)
    dSdt = (delta * M) + (omega * R) - (beta * I * S)/N - (mu * S)
    dIdt = (sigma * E) - ((gamma + mu) * I)
    dRdt = (gamma * I) - ((omega + mu) * R)
    return dMdt, dSdt, dEdt, dIdt, dRdt

y0 = M0, S0, E0, I0, R0
sol = solve_ivp(deriv, [0,100], y0, args=(N, lam, delta, beta, sigma, gamma, omega, mu), t_eval = t)
M, S, E, I, R = sol.y # extract results

# plotting
plt.figure(figsize=(20, 11))
plt.plot(t, M, 'o', label="Immune Infants")
plt.plot(t, S, 'b', label="Suceptible")
plt.plot(t, E, 'y', label="Exposed")
plt.plot(t, I, 'r', label="Infected")
plt.plot(t, R, 'g', label="Recovered")
plt.title("MSEIR Model")
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid(lw = 2, ls = ":")
plt.show()