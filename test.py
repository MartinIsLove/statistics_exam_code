from math import exp, log, log1p, pi

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def exponential_sample(rng, n, rate):
    return -np.log1p(-rng.random(n)) / rate


def normal_sample(rng, n):
    u1, u2 = rng.random((2, (n + 1) // 2))
    radius = np.sqrt(-2 * np.log1p(-u1))
    angle = 2 * pi * u2
    return np.concatenate(
        (radius * np.cos(angle), radius * np.sin(angle))
    )[:n]


def gamma_sample(rng, n, shape, rate, proposal_rate):
    if shape <= 1 or not 0 < proposal_rate < rate:
        raise ValueError("Require shape > 1 and 0 < proposal_rate < rate.")

    delta = rate - proposal_rate
    peak_x = (shape - 1) / delta
    log_peak_x = log(peak_x)
    accepted = []
    proposals = 0

    while len(accepted) < n:
        u, v = rng.random(2)
        y = -log1p(-u) / proposal_rate
        proposals += 1

        if y == 0:
            continue

        log_probability = (
            (shape - 1) * (log(y) - log_peak_x)
            - delta * (y - peak_x)
        )
        if v < exp(log_probability):
            accepted.append(y)

    return np.asarray(accepted), proposals


n = 10_000
rng = np.random.Generator(np.random.PCG64(2026))

exponential = exponential_sample(rng, n, log(2))
normal = normal_sample(rng, n)
gamma_half, proposals_half = gamma_sample(rng, n, 3, 1, 1 / 2)
gamma_third, proposals_third = gamma_sample(rng, n, 3, 1, 1 / 3)
a, _ = gamma_sample(rng, n, 2, 1, 1 / 2)
b, _ = gamma_sample(rng, n, 5, 1, 1 / 5)
beta = a / (a + b)

gamma_reference = stats.gamma(a=3, scale=1)
comparisons = [
    ("Exponential", exponential, stats.expon(scale=1 / log(2))),
    ("Standard normal", normal, stats.norm()),
    ("Gamma(3, 1)", gamma_third, gamma_reference),
    ("Beta(2, 5)", beta, stats.beta(2, 5)),
]

def print_row(test, measure, theoretical, observed, difference="-"):
    print(
        f"{test:<19} {measure:<21} {theoretical:>10} "
        f"{observed:>10} {difference:>11}"
    )


def print_number(test, measure, theoretical, observed, digits=5):
    print_row(
        test, measure,
        f"{theoretical:.{digits}f}",
        f"{observed:.{digits}f}",
        f"{observed - theoretical:+.{digits}f}",
    )


print(f"\nCONFRONTO TEORIA E SIMULAZIONE - {n} campioni per prova\n")
print_row("Distribuzione/prova", "Misura", "Teorico", "Osservato", "Scarto")
print("-" * 75)

for name, sample, reference in comparisons:
    print_number(name, "Media", reference.mean(), sample.mean())
    print_number(name, "Varianza", reference.var(), sample.var(ddof=1))
    d_n = stats.kstest(sample, reference.cdf).statistic
    print_row(name, "D KS (p.p.)", "-", f"{100 * d_n:.3f}")

print("-" * 75)

for label, sample, count, p_theory in [
    ("Gamma theta=1/2", gamma_half, proposals_half, exp(2) / 16),
    ("Gamma theta=1/3", gamma_third, proposals_third, 2 * exp(2) / 27),
]:
    accepted = len(sample)
    p_observed = accepted / count
    expected_proposals = accepted / p_theory

    print_row(
        label, "Proposte totali",
        f"{expected_proposals:.2f}", str(count),
        f"{count - expected_proposals:+.2f}",
    )
    print_row(
        label, "Accettazione",
        f"{p_theory:.2%}", f"{p_observed:.2%}",
        f"{100 * (p_observed - p_theory):+.2f} pp",
    )
    print_number(label, "Proposte/campione", 1 / p_theory, count / accepted, 4)

    d_n = stats.kstest(sample, gamma_reference.cdf).statistic
    print_row(label, "D KS (p.p.)", "-", f"{100 * d_n:.3f}")

print("-" * 75)

waits = np.count_nonzero(exponential <= 2)
p_theory = 0.75
p_observed = waits / len(exponential)
expected_waits = len(exponential) * p_theory

print_row(
    "Esponenziale", "P(attesa <= 2 min)",
    f"{p_theory:.2%}", f"{p_observed:.2%}",
    f"{100 * (p_observed - p_theory):+.2f} pp",
)
print_row(
    "Esponenziale", "Attese <= 2 min",
    f"{expected_waits:.0f}", str(waits),
    f"{waits - expected_waits:+.0f}",
)

print("\nScarto = osservato - teorico, calcolato prima dell'arrotondamento.")
print("p.p. = punti percentuali.")
print("D KS = distanza massima tra CDF empirica e teorica.")


fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7))

for ax, (name, sample, reference) in zip(axes.flat, comparisons):
    left, right = reference.support()
    if not np.isfinite(left):
        left = min(sample.min(), reference.ppf(0.001))
    if not np.isfinite(right):
        right = max(sample.max(), reference.ppf(0.999))

    bins = np.linspace(left, right, 41)
    x = np.linspace(left, right, 600)

    ax.hist(
        sample, bins=bins, density=True,
        color="#4C78A8", alpha=0.48, edgecolor="white", linewidth=0.35,
        label="Sample histogram",
    )
    ax.plot(
        x, reference.pdf(x), color="#B34824", linewidth=2,
        label="Theoretical density",
    )

    ax.set(title=name, xlabel="$x$", ylabel="Density", xlim=(left, right))
    ax.set_ylim(bottom=0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
    ncol=2, frameon=False,
)
fig.tight_layout(rect=(0, 0.08, 1, 1))

with plt.rc_context({"pdf.fonttype": 42}):
    fig.savefig("density_comparison.pdf", bbox_inches="tight")

plt.close(fig)