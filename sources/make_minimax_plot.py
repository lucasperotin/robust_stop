import numpy as np
import matplotlib.pyplot as plt

# Two-period robust example: bar r_1=bar r_2=1, rho=2.
# The paper reports loss factors OPT/ALG, hence values are >= 1.
a = np.linspace(0, 1, 400)
success_regime_high = 1 - a / 3
success_regime_low = (2 * a + 3) / 5
loss_high = 1 / success_regime_high
loss_low = 1 / success_regime_low
loss_worst = np.maximum(loss_high, loss_low)
a_star = 6 / 11
loss_star = 11 / 9

plt.figure(figsize=(7.2, 4.4))
plt.plot(a, loss_high, label=r"regime $r_2\geq 1$: $(1-a/3)^{-1}$")
plt.plot(a, loss_low, label=r"regime $r_2<1$: $5/(2a+3)$")
plt.plot(a, loss_worst, label=r"worst case $\Delta(a)$", linewidth=2)
plt.axvline(a_star, linestyle="--", linewidth=1)
plt.axhline(loss_star, linestyle="--", linewidth=1)
plt.scatter([a_star], [loss_star], zorder=5)
plt.text(a_star + 0.02, loss_star + 0.035, r"$a^*=6/11$, $\Delta=11/9$")
plt.xlabel(r"randomization probability $a$")
plt.ylabel(r"worst-case loss factor $\mathrm{OPT}/\mathrm{ALG}$")
plt.ylim(1.0, 1.72)
plt.xlim(0, 1)
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("minimax_n2.pdf")
plt.savefig("minimax_n2.png", dpi=200)
