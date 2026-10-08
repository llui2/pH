"""Exact stationary dynamics of the six-reaction network of Kosc et al.

Internal species: X1, X2, X3, X4, X1', X2'.
Reactions: X1+F <-> B+X2+W; X1' <-> A+X2';
X2+X2' <-> X3; X3 <-> 2X4; A+X4 <-> X1; B+X4 <-> X1'.
Buffered A=B=W=1, F=exp(drive). All bare forward/reverse rates equal 1.
Dimensionless volume is 4. Currents are exact finite-state stationary averages.
"""
from pathlib import Path
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

S = np.array([
    [-1, 0, 0, 0, 1, 0],
    [1, 0, -1, 0, 0, 0],
    [0, 0, 1, -1, 0, 0],
    [0, 0, 0, 2, -1, -1],
    [0, -1, 0, 0, 0, 1],
    [0, 1, -1, 0, 0, 0],
], dtype=int)
WEIGHTS = np.array([1, 1, 2, 1, 1, 1])
VOLUME = 4
MASSES = (4, 6, 8)
DRIVES = np.linspace(0, 6, 41)
OUT = Path(__file__).with_name("cores.pdf")


def states(m):
    result = []

    def add(part, k, left):
        if k == 5:
            result.append(tuple(part + [left]))
            return
        for count in range(left // WEIGHTS[k] + 1):
            add(part + [count], k + 1, left - count * WEIGHTS[k])

    add([], 0, m)
    return result


def stationary(m, drive):
    configs = states(m)
    index = {n: i for i, n in enumerate(configs)}
    rows, cols, values = [], [], []
    currents = np.zeros((len(configs), 6))

    for i, n in enumerate(configs):
        x1, x2, x3, x4, y1, y2 = n
        forward = (np.exp(drive) * x1, y1, x2 * y2 / VOLUME, x3, x4, x4)
        backward = (x2, y2, x3, x4 * (x4 - 1) / VOLUME, x1, y1)
        currents[i] = np.subtract(forward, backward)
        total = 0.0

        for r in range(6):
            for sign, rate in ((1, forward[r]), (-1, backward[r])):
                if rate <= 0:
                    continue
                dest = tuple(np.array(n) + sign * S[:, r])
                rows.append(index[dest])
                cols.append(i)
                values.append(rate)
                total += rate

        rows.append(i)
        cols.append(i)
        values.append(-total)

    generator = coo_matrix(
        (values, (rows, cols)), shape=(len(configs),) * 2
    ).tolil()
    generator[-1, :] = 1
    rhs = np.zeros(len(configs))
    rhs[-1] = 1
    probabilities = spsolve(generator.tocsc(), rhs)
    return np.asarray(configs), probabilities, probabilities @ currents


assert np.all(WEIGHTS @ S == 0)
assert np.all(S @ np.ones(6) == 0)
assert np.linalg.matrix_rank(S) == 5

n, p, v = stationary(6, 0)
equilibrium = np.array([
    VOLUME ** sum(x) / np.prod([factorial(int(k)) for k in x])
    for x in n
])
equilibrium /= equilibrium.sum()
assert np.max(np.abs(p - equilibrium)) < 1e-11

fig, axes = plt.subplots(1, 2, figsize=(7, 2.75))
for m, color in zip(MASSES, ("#536270", "#28759c", "#bf7440")):
    currents, supply = [], []
    for drive in DRIVES:
        n, p, v = stationary(m, drive)
        assert abs(v.max() - v.min()) < 1e-9
        assert drive * v.mean() >= -1e-10
        currents.append(v.mean())
        supply.append(p @ n[:, 5] / m)

    axes[0].plot(DRIVES, currents, color=color, lw=1.7, label=fr"$m={m}$")
    axes[1].plot(DRIVES, supply, color=color, lw=1.7)

for panel, ax in zip(("a", "b"), axes):
    ax.text(-0.14, 1.04, f"({panel})", transform=ax.transAxes, size=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 6)
    ax.tick_params(labelsize=8)

axes[0].set(xlabel=r"fuel driving $\beta\Delta\mu$", ylabel=r"cycle current $J$", ylim=(0, None))
axes[1].set(xlabel=r"fuel driving $\beta\Delta\mu$", ylabel=r"$\langle n_{X_2^\prime}\rangle/m$", ylim=(0, None))
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.subplots_adjust(left=0.1, bottom=0.21, right=0.98, top=0.91, wspace=0.38)
fig.savefig(OUT)
plt.close(fig)
print(f"Equilibrium and stationary-current checks passed. Figure: {OUT}")
