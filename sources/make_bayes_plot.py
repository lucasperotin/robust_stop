import numpy as np
import matplotlib.pyplot as plt
from scipy.special import exp1, erfc
from scipy.integrate import quad

# Calibration B_R(x) = x M(x)/(1 - x M(x)).

def B_det1(x):
    return x

def M_uniform(x, a=0.5, b=1.5):
    return np.log((b+x)/(a+x))/(b-a)

def B_uniform(x, a=0.5, b=1.5):
    M = M_uniform(x,a,b)
    return x*M/(1-x*M)

def M_exp1(x): # Exp(rate=1), mean 1. This is Gamma(1,1)
    return np.exp(x)*exp1(x)

def B_exp1(x):
    M = M_exp1(x)
    return x*M/(1-x*M)

def M_shifted_halfnormal(x, sigma=0.5):
    a = x + 1.0
    z = a/(np.sqrt(2)*sigma)
    return np.sqrt(np.pi/2)/sigma * np.exp(z*z) * erfc(z)

def B_shifted_halfnormal(x, sigma=0.5):
    M = M_shifted_halfnormal(x, sigma)
    return x*M/(1-x*M)

xs = np.logspace(-2, 1, 300)
plt.figure(figsize=(7.5,4.8))
plt.plot(xs, B_det1(xs), label=r'$\rho\equiv 1$')
plt.plot(xs, B_uniform(xs), label=r'$\rho\sim\mathrm{Unif}[0.5,1.5]$')
plt.plot(xs, B_exp1(xs), label=r'$\rho\sim\mathrm{Exp}(1)$')
plt.plot(xs, B_shifted_halfnormal(xs), label=r'$\rho=1+|N(0,0.5^2)|$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'predicted odd $\bar r=x$')
plt.ylabel(r'Bayesian effective odd $B_R(x)$')
plt.title('Bayesian calibration of predicted odds')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('bayes_calibration.pdf')
plt.savefig('bayes_calibration.png', dpi=200)

# Print table values
xvals = [0.1, 0.5, 1, 2, 5]
print('x det uniform exp shifted_halfnormal')
for x in xvals:
    print(x, B_det1(x), B_uniform(x), B_exp1(x), B_shifted_halfnormal(x))
