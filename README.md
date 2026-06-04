# Ito integral approximation
per textbook problems. Source: _Modeling with Itô Stochastic Differential Equations_ by Edward Allen, pg. 74.

# Background
Exercise 3.11** (Approximation of an Itô Integral).
In this example, the stochastic integral $\int^t_0tW(t)dW(t)$ is considered. The expected value of the integral and the expected value of the square of the integral are estimated using $M=1,000,000$ sample paths. The number of intervals used on $[0,1]$ is $N$ where $N=4,8,16,\dots,256$. The preceding approximate method is used, which for this problem, has the explicit form 

$$\begin{align*}
E((I(f))^\alpha)&=E\left(\int^1_0tW(t)dW(t)\right)^\alpha\\
&\approx\frac{1}{M}\sum^M_{j=1}\left(\sum^{N-1}_{i=0}t_i^{(N)}W^{(j)}(t_i^{(N)})(W^{(j)}(t^{(N)}_{t+1})-W^{(j)}(t^{(N)}_i))\right)^\alpha\quad(2)\\
\end{align*}$$

for $\alpha=1$ or $2$ and $t_i^{(N)}=i/N$ for $i=0,1,2,\dots,N$. Notice that $(W^{(j)}(t^{(N)}_{i+1})-W^{(j)}(t_i^{(N)}))=\eta_i^{(j)}/\sqrt{N}$ where $\eta_i^{(j)}\sim N(0,1)$ and also that $(W^{(j)}(t^{(N)}_{i})-W^{(j)}(t_{i-1}^{(N)}))=\eta_{i-1}^{(j)}/\sqrt{N}$. A computer program that performs this calculation is listed at the end of this chapter. The calculational results are given in Table 3.2 for $M=1,000,000$ sample paths. Notice that the results improve as $N$ increases. The exact values are $E(I(f))=0$ and $E((I(f))^2)=0.25.$

$$\begin{array}{ccc}
\hline
\text{Value of N}&|E(I(f))|&|E((I(f))^2)|\\
\hline
\color{red}{2^2}&\color{red}{0.00107}&\color{red}{0.14071}\\
2^3&0.00012&0.19151\\
2^4&0.00069&0.21906\\
2^5&0.00038&0.23508\\
2^6&0.00007&0.24209\\
2^7&0.00001&0.24486\\
2^8&0.00002&0.24821\\
\hline
\end{array}$$

The moments of the stochastic process, namely $X_t = \int_0^t sW_s \,\mathrm{d}W_s$, can be determined analytically and one has $\mathbb{E}[X_t] = 0$, since $W_s$ and $\mathrm{d}W_s$ are independent and have zero mean, while Itô's isometry gives 

$$\mathrm{Var}[X_t] = \mathbb{E}[X_t^2] = \mathbb{E}\left[\int_0^t s^2W_s^2 \,\mathrm{d}s\right] = \int_0^t s^2\,\mathbb{E}[W_s^2] \,\mathrm{d}s = \int_0^t s^3 \,\mathrm{d}s = \frac{1}{4}t^4$$

which is equal to $1/4$ when $t=1$.

```
import numpy as np

T = 1.0 
N = 1000        # no. of out steps t (time) reduced for speed of code output
dt = T / N
nbSim = 10000   # no. of our simulations
Ito_sum = 0.0

# our time grid at the left endpoints here
t = np.linspace(0, T - dt, N) 

for j in range(nbSim):
    # now, we generate the brownian increments 
    dW = np.sqrt(dt) * np.random.randn(N)
    # now, we compute the brownian motion paths, starting at 0
    W = np.cumsum(dW)
    
    # now, we use 0 to get the value of W at the left endpoints here
    # and drop the last element
    W_left = np.insert(W, 0, 0)[:-1]
    
    # now, the itô integral is therefore: sum of (t_i * W_i * dW_i)
    ito_trajectory = np.sum(t * W_left * dW)
    
    # now, we accumulate the square of the integral here
    Ito_sum += ito_trajectory ** 2

# now, we calculate the expected value 
E_Ito_squared = Ito_sum / nbSim
print("Expected value of the squared Itô integral is: {:.6f}".format(E_Ito_squared))
## Expected value of the squared Itô integral is: 0.236472
```

Then, to simulate the paths:

```
import numpy as np
import matplotlib.pyplot as plt

T = 1.0 
N = 1000        
dt = T / N
nbSim = 10     

# our time grid 
t = np.linspace(0, T - dt, N) 

plt.figure(figsize=(10, 6))

for j in range(nbSim):
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)
    W_left = np.insert(W, 0, 0)[:-1]
    
    # now, we calculate our running integral over time using cumsum
    # which gives us the value of the integral at every single time step
    ito_path = np.cumsum(t * W_left * dW)
    
    # now, we plot our simulation's path
    plt.plot(np.linspace(0, T, N), ito_path, label=f'Path {j+1}')

plt.title(r'Simulated Paths of the Itô Integral $\int_0^t s W_s \, dW_s$', fontsize=14)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('Integral Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black', lw=1) # Adds our baseline at 0
plt.savefig('simulateditointegral.png', dpi=300, bbox_inches='tight')
plt.show()
```
# Final result
<img width="636" height="413" alt="itointegral" src="https://github.com/user-attachments/assets/4138dacf-cebe-48fe-8a0f-941dbae340ce" />







