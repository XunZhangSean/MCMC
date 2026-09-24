"""
CS 6786 Homework 1 - Question 4

Name: Xun Zhang

This script runs RWMH and NUTS on three 2D target distributions and
creates the plots used in the homework report.

I used BlackJAX functions for the MCMC algorithms and focused on comparing
how different samplers explore different posterior shapes.
"""

from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp
import blackjax
import numpy as np
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import corner

# Student information
NAME = "Xun Zhang"

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT / "results"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
SEED, N, BURN = 20260923, 12000, 2000
SIGMA = jnp.array([[0.5, 0.25], [0.25, 0.35]])
MU1, MU2 = jnp.array([-1., -2.]), jnp.array([4., 4.])
BLUE, RED, GRAY = "#2266aa", "#bf4d3a", "#555555"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def gaussian(z):
    return jsp.stats.multivariate_normal.logpdf(z, jnp.zeros(2), SIGMA)


def components(z):
    return jnp.stack([
        jnp.log(0.4) + jsp.stats.multivariate_normal.logpdf(z, MU1, SIGMA),
        jnp.log(0.6) + jsp.stats.multivariate_normal.logpdf(z, MU2, 1.5*jnp.eye(2))
    ])


def mixture(z):
    return jsp.special.logsumexp(components(z))


def funnel(z):
    # exp(z1/2) is the variance, so the standard deviation is exp(z1/4).
    return (jsp.stats.norm.logpdf(z[0], 0., 2.)
            + jsp.stats.norm.logpdf(z[1], 0., jnp.exp(z[0]/4.)))


TARGETS = {"gaussian": gaussian, "mixture": mixture, "funnel": funnel}
STARTS = {"gaussian": [0., 0.], "mixture": [-1., -2.], "funnel": [0., 0.]}
SCALES = {"gaussian": [0.1, 0.5, 1.5], "mixture": [1., 3., 6.],
          "funnel": [0.2, 1., 3.]}
ACCEPT_TARGETS = [0.7, 0.9, 0.99]


def run_chain(key, algorithm, state, count):
    def step(state, key):
        state, info = algorithm.step(key, state)
        return state, (state.position, info)
    return jax.jit(lambda s, ks: jax.lax.scan(step, s, ks))(
        state, jax.random.split(key, count))


def independent_samples(name, rng, n):
    if name == "gaussian":
        return rng.multivariate_normal(np.zeros(2), np.asarray(SIGMA), n)
    if name == "mixture":
        first = rng.uniform(size=n) < 0.4
        z = np.empty((n, 2))
        z[first] = rng.multivariate_normal(np.asarray(MU1), np.asarray(SIGMA), first.sum())
        z[~first] = rng.multivariate_normal(np.asarray(MU2), 1.5*np.eye(2), (~first).sum())
        return z
    z1 = rng.normal(0., 2., n)
    z2 = rng.normal(size=n) * np.exp(z1/4.)
    return np.column_stack([z1, z2])


def diagnostics(name, samples):
    post = samples[BURN:]
    bulk = [float(az.ess(post[:, i][None, :], method="bulk")) for i in range(2)]
    tail = [float(az.ess(post[:, i][None, :], method="tail")) for i in range(2)]
    d = {"mean": post.mean(0).tolist(), "covariance": np.cov(post.T).tolist(),
         "ess_bulk": bulk, "ess_tail": tail}
    if name == "mixture":
        # Use the more likely Gaussian component to check whether the chain moves between modes.
        logcomp = np.asarray(jax.vmap(components)(jnp.asarray(post)))
        in_first = logcomp[:, 0] > logcomp[:, 1]
        switches = int(np.count_nonzero(np.diff(in_first.astype(int))))
        d.update(first_region=float(in_first.mean()), switches=switches,
                 region_ess=(float(az.ess(in_first.astype(float)[None, :], method="bulk"))
                             if switches else 0.))
    return d


def hello_world():
    rng = np.random.default_rng(SEED)
    observed = jnp.asarray(rng.normal(10., 20., 1000))
    def logdensity(z):
        return z[1] + jnp.sum(jsp.stats.norm.logpdf(observed, z[0], jnp.exp(z[1])))
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, params), _ = warmup.run(jax.random.key(SEED), jnp.array([1., 1.]), num_steps=1000)
    _, (samples, _) = run_chain(jax.random.key(SEED+1), blackjax.nuts(logdensity, **params), state, 2000)
    samples = np.asarray(samples)
    return {"description": "BlackJAX quickstart normal location and log-scale example",
            "n_observed": 1000, "warmup": 1000, "retained": 2000,
            "posterior_mean_location": float(samples[:, 0].mean()),
            "posterior_mean_scale": float(np.exp(samples[:, 1]).mean())}


def experiment():
    records, chains = [], {}
    for ti, (name, target) in enumerate(TARGETS.items()):
        initial = jnp.array(STARTS[name])
        for method in ["RWMH", "NUTS"]:
            values = SCALES[name] if method == "RWMH" else ACCEPT_TARGETS
            for ci, value in enumerate(values):
                seed = SEED + 100 + 100*ti + 10*(method == "NUTS") + ci
                key0, key1 = jax.random.split(jax.random.key(seed))
                start_time = time.perf_counter()
                rec = {"target": name, "method": method, "trial": ci+1,
                       "seed": seed, "initial": STARTS[name], "iterations": N,
                       "burn_in": BURN, "retained": N-BURN}
                if method == "RWMH":
                    # For RWMH, this parameter controls the proposal step size.
                    alg = blackjax.normal_random_walk(target, jnp.full(2, value))
                    _, (positions, info) = run_chain(key1, alg, alg.init(initial), N)
                    samples = np.asarray(positions)
                    rec.update(scale=value, acceptance=float(np.asarray(info.is_accepted)[BURN:].mean()),
                               divergences=None)
                else:
                    warmup = blackjax.window_adaptation(
                        blackjax.nuts, target, target_acceptance_rate=value,
                        is_mass_matrix_diagonal=True, max_num_doublings=10)
                    (state, params), history = warmup.run(key0, initial, num_steps=BURN)
                    alg = blackjax.nuts(target, **params)
                    _, (positions, info) = run_chain(key1, alg, state, N-BURN)
                    samples = np.concatenate([np.asarray(history.state.position), np.asarray(positions)])
                    nsteps = np.asarray(info.num_integration_steps)
                    rec.update(target_acceptance=value, step_size=float(params["step_size"]),
                               inverse_mass=np.asarray(params["inverse_mass_matrix"]).tolist(),
                               acceptance=float(np.asarray(info.acceptance_rate).mean()),
                               divergences=int(np.asarray(info.is_divergent).sum()),
                               mean_leapfrog=float(nsteps.mean()),
                               depth_limit_hits=int((np.asarray(info.num_trajectory_expansions) >= 10).sum()))
                rec["seconds_including_compile"] = time.perf_counter()-start_time
                assert samples.shape == (N, 2) and np.isfinite(samples).all()
                rec.update(diagnostics(name, samples))
                key = f"{name}_{method}_{ci+1}"
                chains[key] = samples
                records.append(rec)
                print(json.dumps(rec), flush=True)
                # Save intermediate results so the experiments can be reused.
                (DATA / "diagnostics.json").write_text(json.dumps(records, indent=2))
                np.save(DATA / f"{key}.npy", samples)
    return records, chains


def choose_runs(records):
    chosen = {}
    for name in TARGETS:
        for method in ["RWMH", "NUTS"]:
            candidates = [r for r in records if r["target"] == name and r["method"] == method]
            # Prefer NUTS runs without divergences when selecting a representative run.
            if method == "NUTS":
                mindiv = min(r["divergences"] for r in candidates)
                candidates = [r for r in candidates if r["divergences"] == mindiv]
            def score(r):
                ess = min(r["ess_bulk"])
                return min(ess, r["region_ess"]) if name == "mixture" else ess
            chosen[f"{name}_{method}"] = max(candidates, key=score)
    return chosen


def figures(records, chains, chosen):
    bounds = {"gaussian": [(-3, 3), (-3, 3)], "mixture": [(-4, 8), (-5, 8)],
              "funnel": [(-7, 7), (-8, 8)]}
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.35), constrained_layout=True)
    for ax, (name, target) in zip(axes, TARGETS.items()):
        (lo, hi), (lo2, hi2) = bounds[name]
        x, y = np.meshgrid(np.linspace(lo, hi, 250), np.linspace(lo2, hi2, 250))
        logp = np.asarray(jax.vmap(target)(jnp.asarray(np.column_stack([x.ravel(), y.ravel()])))).reshape(x.shape)
        levels = np.linspace(logp.max()-10., logp.max()-0.05, 9)
        ax.contour(x, y, logp, levels=levels, cmap="viridis", linewidths=1.)
        ax.set(xlabel=r"$z_1$", ylabel=r"$z_2$", title=name.capitalize())
    fig.savefig(FIG / "target_contours.pdf", bbox_inches="tight")
    fig.savefig(FIG / "target_contours.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    for ti, name in enumerate(TARGETS):
        fig, axes = plt.subplots(3, 2, figsize=(11.4, 8.), sharex=True, constrained_layout=True)
        for mi, method in enumerate(["RWMH", "NUTS"]):
            rs = [r for r in records if r["target"] == name and r["method"] == method]
            for row, rec in enumerate(rs):
                ax = axes[row, mi]
                s = chains[f"{name}_{method}_{rec['trial']}"]
                ax.plot(np.arange(1, N+1), s[:, 0], color=BLUE, lw=.35, alpha=.8, label=r"$z_1$")
                ax.plot(np.arange(1, N+1), s[:, 1], color=RED, lw=.35, alpha=.7, label=r"$z_2$")
                ax.axvspan(0, BURN, color="gray", alpha=.15)
                ax.axvline(BURN, color="black", ls=":", lw=.8)
                config = f"s={rec['scale']:g}" if method == "RWMH" else f"target acceptance={rec['target_acceptance']:g}"
                sel = " [selected]" if rec["trial"] == chosen[f"{name}_{method}"]["trial"] else ""
                ax.set_title(f"{method}: {config}{sel}", fontsize=10)
                ax.set_ylabel("Value")
                if row == 2:
                    ax.set_xlabel("Iteration")
                if row == 0:
                    ax.legend(loc="upper right", fontsize=8, ncol=2)
        fig.suptitle(f"{name.capitalize()}: all trials (gray = discarded warmup)", fontsize=13)
        fig.savefig(FIG / f"traces_{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        rng = np.random.default_rng(SEED + 1000 + ti)
        reference = independent_samples(name, rng, N-BURN)
        np.save(DATA / f"{name}_iid.npy", reference)
        if name == "mixture":
            cp = np.asarray(jax.vmap(components)(jnp.asarray(reference)))
            (DATA / "mixture_reference_region.json").write_text(json.dumps({
                "first_region": float((cp[:, 0] > cp[:, 1]).mean())}))
        for method, color in [("RWMH", BLUE), ("NUTS", RED)]:
            rec = chosen[f"{name}_{method}"]
            samples = chains[f"{name}_{method}_{rec['trial']}"][BURN:]
            opts = dict(labels=[r"$z_1$", r"$z_2$"], bins=45, range=bounds[name],
                        plot_datapoints=False, plot_density=False,
                        levels=[0.5, 0.9], smooth=1., smooth1d=None,
                        hist_kwargs={"density": True}, label_kwargs={"fontsize": 13})
            fig = corner.corner(reference, color=GRAY, **opts)
            corner.corner(samples, fig=fig, color=color, **opts)
            fig.suptitle(f"{name.capitalize()} / {method}", fontsize=14, y=.98)
            fig.axes[1].legend(handles=[Line2D([], [], color=GRAY, label="Direct iid"),
                                      Line2D([], [], color=color, label=method)],
                               frameon=False, loc="center", fontsize=12)
            fig.savefig(FIG / f"corner_{name}_{method}.pdf", bbox_inches="tight")
            plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="Reuse the saved trial arrays")
    args = parser.parse_args()
    versions = {"jax": jax.__version__, "blackjax": blackjax.__version__,
                "numpy": np.__version__, "arviz": az.__version__, "corner": corner.__version__}
    (DATA / "versions.json").write_text(json.dumps(versions, indent=2))
    if args.plot_only:
        records = json.loads((DATA / "diagnostics.json").read_text())
        chosen = choose_runs(records)
        chains = {f"{r['target']}_{r['method']}_{r['trial']}": np.load(
            DATA / f"{r['target']}_{r['method']}_{r['trial']}.npy") for r in records}
        figures(records, chains, chosen)
        print("Regenerated figures from saved samples.", flush=True)
        return
    print("Checking the BlackJAX quickstart example...", flush=True)
    hello = hello_world()
    print(json.dumps(hello), flush=True)
    (DATA / "hello_world.json").write_text(json.dumps(hello, indent=2))
    records, chains = experiment()
    chosen = choose_runs(records)
    (DATA / "selected.json").write_text(json.dumps(chosen, indent=2))
    figures(records, chains, chosen)
    print("Finished. Figures and numerical diagnostics saved.", flush=True)


if __name__ == "__main__":
    main()
