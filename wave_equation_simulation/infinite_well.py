#ASsumption of infinite well:
import numpy as np
import matplotlib.pyplot as plt
pi =  3.1415
n=1
L = 1
def P(x):
    return 2/L*(np.sin(n*pi*x/L))**2


#plot by chatgpt
x = np.linspace(0,L,500)
plt.figure(figsize=(8, 4))
plt.plot(x, P(x), label=f"n = {n}")
plt.xlabel("Position (x)")
plt.ylabel("Probability Density P(x)")
plt.title("Probability Density for a Particle in an Infinite Potential Well")
plt.legend()
plt.grid(True)
plt.show()