import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import lambertw


def x_star_discrete(rho, n):
    if rho == 1:
        return 1.0

    def f(x):
        return x * (1 + (x - 1) / (n - 1)) ** (n - 1) - rho**2

    return brentq(f, 1.0, rho**2)


def discrete_values(rho, n=10):
    x = x_star_discrete(rho, n)
    theta = x / rho
    loss_opt = rho**2 / x
    loss_naive = (1 + (rho - 1) / (n - 1)) ** (n - 1)
    return theta, loss_opt, loss_naive


def discrete_limit_values(rho):
    x = float(lambertw(np.e * rho**2).real)
    theta = x / rho
    loss_opt = rho**2 / x
    loss_naive = np.exp(rho - 1)
    return theta, loss_opt, loss_naive


def poisson_values(rho):
    if rho == 1:
        return 1.0, 1.0, 1.0
    a = 2 * np.log(rho) / (rho**2 - 1)
    theta = rho * a
    loss_opt = (1 / a) * np.exp(a - 1)
    loss_naive = max(
        rho * np.exp(1 / rho - 1),
        (1 / rho) * np.exp(rho - 1),
    )
    return theta, loss_opt, loss_naive


rhos = np.linspace(1, 10, 400)
disc10 = np.array([discrete_values(r, 10) for r in rhos])
discinf = np.array([discrete_limit_values(r) for r in rhos])
pois = np.array([poisson_values(r) for r in rhos])

plt.figure(figsize=(7.5, 4.8))
plt.plot(rhos, disc10[:, 1], label=r"optimized, $n=10$")
plt.plot(rhos, disc10[:, 2], label=r"naive, $n=10$")
plt.plot(rhos, discinf[:, 1], "--", label=r"optimized, $n=\infty$")
plt.plot(rhos, discinf[:, 2], "--", label=r"naive, $n=\infty$")
plt.yscale("log")
plt.xlabel(r"multiplicative error bound $\rho$")
plt.ylabel(r"loss factor $\mathrm{OPT}/\mathrm{ALG}$")
plt.title("Discrete predicted-odds model")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("discrete_ratios.pdf")
plt.savefig("discrete_ratios.png", dpi=200)

plt.figure(figsize=(7.5, 4.8))
plt.plot(rhos, pois[:, 1], label=r"optimized Poisson target")
plt.plot(rhos, pois[:, 2], label=r"naive target")
plt.yscale("log")
plt.xlabel(r"multiplicative error bound $\rho$")
plt.ylabel(r"loss factor $\mathrm{OPT}/\mathrm{ALG}$")
plt.title("Continuous Poisson predicted-intensity model")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("poisson_ratios.pdf")
plt.savefig("poisson_ratios.png", dpi=200)

plt.figure(figsize=(7.5, 4.8))
plt.plot(rhos, disc10[:, 0], label=r"discrete robust target, $n=10$")
plt.plot(rhos, discinf[:, 0], "--", label=r"large-$n$ limit of discrete bound")
plt.plot(rhos, pois[:, 0], label=r"Poisson robust target")
plt.axhline(1.0, color="black", linewidth=1, linestyle=":", label=r"naive target")
plt.xlabel(r"multiplicative error bound $\rho$")
plt.ylabel(r"target level $\theta$")
plt.title("Optimized robust target levels")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("targets.pdf")
plt.savefig("targets.png", dpi=200)

table_rhos = [1, 1.25, 1.5, 2, 3, 5, 10]
print("DISCRETE")
print("rho theta10 Lambda10 B10 thetainf Lambdainf Binf")
for r in table_rhos:
    t10, l10, b10 = discrete_values(r, 10)
    ti, li, bi = discrete_limit_values(r)
    print(f"{r:g} {t10:.3f} {l10:.3f} {b10:.3f} {ti:.3f} {li:.3f} {bi:.3f}")

print("POISSON")
print("rho theta Lambda B")
for r in table_rhos:
    t, l, b = poisson_values(r)
    print(f"{r:g} {t:.3f} {l:.3f} {b:.3f}")
