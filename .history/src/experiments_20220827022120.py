import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from envs import Environment
from envs import NoiseGenerator
from envs import StochasticContextGenerator as CtxGenerator
from envs import Example2ContextGenerator as Ex2CtxGenerator
from envs import Example1ContextGenerator as Ex1CtxGenerator

from policies import Roful

from utils import MetricAggregator
from utils import StateFactory




def example1_scenario(
        state_factory, k=3, d=300, t=1000, prior_var=1, prior_mu=0.0, noise_sd=10.0,
        thin_thresh=2.0,
        const_infl=5.0
):
    param = state_factory().normal(prior_mu, prior_var ** 0.5, d)
    ctx_gen = Ex1CtxGenerator(d, np.sign(noise_sd - (prior_var ** 0.5)))
    noise_gen = NoiseGenerator.gaussian_noise(noise_sd, state=state_factory())

    env = Environment(param, ctx_gen, noise_gen)

    algs = {
        "TS-Bayes": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=1.0),
        # "OFUL": Roful.oful(
        #     d,
        #     alpha = 1,
        #     radius=Roful.radius_inflation()),
        # "TS-2": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=5.0),
        "TS-Freq": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.radius_inflation(),
        ),
        "TS-Improved": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.conditional_inflation(const_infl, thin_thresh=thin_thresh),
        ),
        "TS-Thinness": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.dynamic_inflation(),
        ),
    
        "TS-ThinnessDirected": Roful.thin_dirt(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=1.0,
        ),
    
    }

    for i in range(t):
        ctx = env.next()

        for alg in algs.values():
            idx = alg.choose_arm(ctx)
            fb = env.get_feedback(idx)
            alg.update(fb)

        if i % 100 == 0:
            print(i)

    return {name: (alg.metrics.regrets, alg.thinnesses) for name, alg in algs.items()}

def example2_scenario(
        state_factory, k=3, d=300, t=1000, prior_var=1.0, prior_mu=10.0, noise_sd=1.0,
        thin_thresh=2.0,
        const_infl=5.0
):
    param = state_factory().normal(prior_mu, prior_var ** 0.5, d)
    if k != 3:
        print('*** Warning ***, k should be 3 for fixed action case')
    ctx_gen = Ex2CtxGenerator(d)
    noise_gen = NoiseGenerator.gaussian_noise(noise_sd, state=state_factory())

    env = Environment(param, ctx_gen, noise_gen)

    algs = {
        "TS-Bayes": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=1.0),
        #"OFUL": Roful.oful(
        #    d,
        #    alpha = 1,
        #    radius=Roful.radius_inflation()),
        # "TS-2": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=5.0),
        "TS-Freq": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.radius_inflation(),
        ),
        "TS-Improved": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.conditional_inflation(const_infl, thin_thresh=thin_thresh),
        ),
        "TS-Thinness": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.dynamic_inflation(),
        ),

    
        "TS-ThinnessDirected": Roful.thin_dirt(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=1.0,
        ),
    
    }

    for i in range(t):
        ctx = env.next()

        for alg in algs.values():
            idx = alg.choose_arm(ctx)
            fb = env.get_feedback(idx)
            alg.update(fb)

        if i % 100 == 0:
            print(i)

    return {name: (alg.metrics.regrets, alg.thinnesses) for name, alg in algs.items()}


def russo_scenario(
    state_factory, k=100, d=100, t=1000, prior_var=10.0, arm_bound=0.1,
        prior_mu=0.0, noise_sd=1.0,
        thin_thresh=2.0,
        const_infl=5.0
):
    param = state_factory().normal(prior_mu, prior_var ** 0.5, d)
    ctx_gen = CtxGenerator.uniform_entries(k, d, arm_bound / d ** 0.5, state=state_factory())
    noise_gen = NoiseGenerator.gaussian_noise(noise_sd, state=state_factory())

    env = Environment(param, ctx_gen, noise_gen)
    algs = {
        "TS-Bayes": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=1.0),
        # "TS-2": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=5.0),
        #"OFUL": Roful.oful(
        #    d,
        #    alpha=1,
        #    radius=Roful.radius_inflation()),
        "TS-Freq": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.radius_inflation(),
        ),
        "TS-Improved": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.conditional_inflation(const_infl, thin_thresh=thin_thresh),
        ),
    }

    for i in range(t):
        ctx = env.next()

        for alg in algs.values():
            idx = alg.choose_arm(ctx)
            fb = env.get_feedback(idx)
            alg.update(fb)

        if i % 100 == 0:
            print(i)

    return {name: (alg.metrics.regrets, alg.thinnesses) for name, alg in algs.items()}



def run_experiments(n, d, k, t, s, prior_mu=0.0, prior_sd=10.0, noise_sd=1.0,
                    thin_thresh=2.0,
                    const_infl=5.0,
                    sim=0,
                    ):
    state_factory = StateFactory(s + 1)

    regrets = defaultdict(MetricAggregator)
    cumregrets = defaultdict(MetricAggregator)
    thinnesses = defaultdict(MetricAggregator)
    for i in range(n):
        print(f"Running experiment {i}...")
        if sim == 1:  # Run Example 1
            print('Run Example 1')
            results = example1_scenario(
                d=d, k=k, t=t, state_factory=state_factory,
                prior_var=prior_sd ** 2,
                prior_mu=prior_mu,
                noise_sd=noise_sd,
                thin_thresh=thin_thresh,
                const_infl=const_infl)
        else:
            if sim == 2:  # Run Example 2
                print('Run Example 2')
                results = example2_scenario(
                    d=d, k=k, t=t, state_factory=state_factory,
                    prior_var=prior_sd ** 2,
                    prior_mu=prior_mu,
                    noise_sd=noise_sd,
                    thin_thresh=thin_thresh,
                    const_infl=const_infl)
            else:  # Run Russo Scenario
                print('Run Russo Scenario')
                results = russo_scenario(
                    d=d, k=k, t=t, state_factory=state_factory,
                    prior_var=prior_sd ** 2,
                    prior_mu=prior_mu,
                    noise_sd=noise_sd,
                    thin_thresh=thin_thresh,
                    const_infl=const_infl
                )

        for name, (regret, thinness) in results.items():
            regrets[name].aggregate(regret)
            cumregrets[name].aggregate(np.cumsum(regret))
            thinnesses[name].aggregate(thinness)

    metrics = {
        "regret": regrets,
        "cumregret": cumregrets,
        "thinnesses": thinnesses,
    }
    labels = {
        "regret": "Instantaneous Regret",
        "cumregret": "Cumulative Regret",
        "thinnesses": "Thinness",
    }
    output = pd.DataFrame()
    for name, metric in metrics.items():
        plt.clf()
        for alg, agg in metric.items():
            agg.plot(plt, alg)

            mean, se = agg.get_mean_se()
            nm = alg+'_'+name
            output[nm+'_mean'] = mean
            output[nm+'_se'] = se

        plt.xlabel("Time")
        plt.ylabel(labels[name])

        plt.legend()
        plt.savefig(f"plots/{name}-{n}-{d}-{k}-{t}-{sim}-{prior_mu}-{prior_sd}-{noise_sd}-{thin_thresh}-{const_infl}.pdf")
        output.to_csv(f"plots/{name}-{n}-{d}-{k}-{t}-{sim}-{prior_mu}-{prior_sd}-{noise_sd}-{thin_thresh}-{const_infl}.csv", index=False)
        #plt.show()


def __main__():
    parser = argparse.ArgumentParser(
        description="Run simulations for various ROFUL algorithms."
    )

    parser.add_argument("-n", type=int, help="number of iterations", default=20)
    parser.add_argument("-k", type=int, help="number of actions", default=100)
    parser.add_argument("-d", type=int, help="dimension", default=50)
    parser.add_argument("-t", type=int, help="time horizon", default=10000)
    parser.add_argument("-s", type=int, help="random seed", default=1)
    parser.add_argument("-pm", type=float, help="prior mu", default=0.0)
    parser.add_argument("-psd", type=float, help="prior standard deviation", default=1.0)
    parser.add_argument("-nsd", type=float, help="noise standard deviation", default=1.0)
    parser.add_argument("-th", type=float, help="threshold for thinness based inflation", default=2.0)
    parser.add_argument("-inf", type=float, help="inflation used when large thinness", default=5.0)
    parser.add_argument("-sim", type=int, help="0: russo scenario, 1: example 1, 2: example 2", default=0)

    args = parser.parse_args()

    run_experiments(n=args.n, d=args.d, k=args.k, t=args.t, s=args.s,
                    prior_mu=args.pm,
                    prior_sd=args.psd,
                    noise_sd=args.nsd,
                    thin_thresh=args.th,
                    const_infl=args.inf,
                    sim=args.sim)


if __name__ == "__main__":
    __main__()

# PYTHONPATH=src python -m experiments -sim 1 -k 3 -d 300 -t 1000 -pm 0 -psd 1 -nsd 10
# PYTHONPATH=src python -m experiments -sim 2 -k 3 -d 300 -t 1000 -pm 10 -psd 1 -nsd 1 
