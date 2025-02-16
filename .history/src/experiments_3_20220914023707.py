import argparse

from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from utils import MetricAggregator
from utils import StateFactory

from plotting import *
from examples import *

import time



def run_experiments_prior_sd(n, d_min, d_max, d_gap, k, t, s, prior_mu=0.0, prior_sd=10.0, noise_sd=1.0, thin_thresh=2.0, const_infl=5.0, sim=0, gamma=1):
    state_factory = StateFactory(s)
    d_list = np.arange(d_min, d_gap + d_max, d_gap)
    prior_sd_list = np.arange(2, 12, 2)
    outputs = []
    outputs_last = []
    for d in d_list:
        t = np.int(np.ceil(d / gamma))
        for prior_sd in prior_sd_list:
            regrets = defaultdict(MetricAggregator)
            cumregrets = defaultdict(MetricAggregator)
            thinnesses = defaultdict(MetricAggregator)
            errors = defaultdict(MetricAggregator)
            lambda_maxs = defaultdict(MetricAggregator)
            lambda_mins = defaultdict(MetricAggregator)
            log_maxs_over_mins = defaultdict(MetricAggregator)
            lambdas_second= defaultdict(MetricAggregator)
            lambdas_third = defaultdict(MetricAggregator)


            for i in range(n):
                print(f"Running experiment {i} for dim {d} and sd {prior_sd}...")
                if sim == 1:
                    print('Run Example 1')
                    results = example1_scenario(d=d, k=k, t=t, state_factory=state_factory, prior_var=prior_sd**2, prior_mu=prior_mu, noise_sd=noise_sd, thin_thresh=thin_thresh, const_infl=const_infl)

                elif sim == 2:
                    print('Run Example 2')
                    results = example2_scenario(d=d, k=k, t=t, state_factory=state_factory, prior_var=prior_sd**2, prior_mu=prior_mu, noise_sd=noise_sd, thin_thresh=thin_thresh, const_infl=const_infl)

                elif sim == 3:
                    print('Run Russo Scenario for all methods')
                    results = russo_scenario_all(d=d, k=k, t=t, state_factory=state_factory, prior_var=prior_sd**2, prior_mu=prior_mu, noise_sd=noise_sd, thin_thresh=thin_thresh, const_infl=const_infl)

                else:
                    print('Run Russo Scenario')
                    results = russo_scenario(d=d, k=k, t=t, state_factory=state_factory, prior_var=prior_sd**2, prior_mu=prior_mu, noise_sd=noise_sd, thin_thresh=thin_thresh, const_infl=const_infl)

                for name, (regret, thinness, error, lambda_max, lambda_min, lambda_second, lambda_third) in results.items():
                    regrets[name].aggregate(regret)
                    cumregrets[name].aggregate(np.cumsum(regret))
                    thinnesses[name].aggregate(thinness)
                    errors[name].aggregate(error / d)
                    lambda_maxs[name].aggregate(lambda_max)
                    lambda_mins[name].aggregate(lambda_min)
                    log_maxs_over_mins[name].aggregate(np.log(lambda_max) / lambda_min)
                    lambdas_second[name].aggregate(lambda_second)
                    lambdas_third[name].aggregate(lambda_third)



            metrics = {"regret": regrets, "cumregret": cumregrets, "thinnesses": thinnesses, "errors": errors, "lambda_maxs": lambda_maxs, "lambda_mins": lambda_mins, "log_maxs_over_mins": log_maxs_over_mins, "lambdas_second": lambdas_second, "lambdas_third": lambdas_third}

            labels = {"regret": "Instantaneous Regret", "cumregret": "Cumulative Regret", "thinnesses": "Thinness", "errors": "Normalized Error", "lambda_maxs": "Maximal Eigenvalues", "lambda_mins": "Minimal Eigenvalues", "log_maxs_over_mins": "Log Max / Min", "lambdas_second": "Second Largest Eigenvalues", "lambdas_third": "Third Largest Eigenvalues"}

            output = pd.DataFrame()
            output_last = pd.DataFrame()
            for name, metric in metrics.items():
                for alg, agg in metric.items():
                    mean, se = agg.get_mean_se()
                    nm = alg + '_' + name
                    output['d'] = d
                    output['prior_sd'] = prior_sd
                    output[nm + '_mean'] = mean
                    output[nm + '_se'] = se
                    output_last['d'] = d
                    output_last['prior_sd'] = prior_sd
                    output_last[nm + '_mean'] = [mean[-1]]
                    output_last[nm + '_se'] = [se[-1]]
            outputs.extend([output])
            outputs_last.extend([output_last])
    outputs = pd.concat(outputs)
    outputs_last = pd.concat(outputs_last)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    outputs.to_csv(f"plots/all-unit-{timestr}-{n}-{d_min}-{d_max}-{d_gap}-{k}-{t}-{sim}-{prior_mu}-{prior_sd}-{noise_sd}-{thin_thresh}-{const_infl}-{gamma}.csv", index=False)


    outputs_last.to_csv(f"plots/all-unit-last-{timestr}-{n}-{d_min}-{d_max}-{d_gap}-{k}-{t}-{sim}-{prior_mu}-{prior_sd}-{noise_sd}-{thin_thresh}-{const_infl}-{gamma}.csv", index=False)


    for name, metric in metrics.items():
        plt.clf()
        for alg, agg in metric.items():
            for d in d_list:
                nm = alg + '_' + name
                output_ = outputs_last.loc[outputs_last['d'] == d]
                mean = output_[nm + '_mean']
                se = output_[nm + '_se']
                x = output_['prior_sd']
                scale = 2.0
                lower = mean - scale * se
                upper = mean + scale * se
                plt.fill_between(x, lower, upper, alpha=0.2)
                plt.plot(x, mean, label=alg + '_dim_' + str(d))
        plt.xlabel("prior_sd")
        plt.ylabel(labels[name])
        plt.legend()
        plt.savefig(
            f"plots/{timestr}-unit-{'last_sd'}-{name}-{n}-{d_min}-{d_max}-{d_gap}-{k}-{t}-{sim}-{prior_mu}-{prior_sd}-{noise_sd}-{thin_thresh}-{const_infl}-{gamma}.pdf")


def run_experiments_k(
    n, d_min, d_max, d_gap, k, t, s, prior_mu=0.0, prior_sd=10.0, noise_sd=1.0,
    thin_thresh=2.0,
    const_infl=5.0,
    sim=0,
    gamma=1,
):
    state_factory = StateFactory(s)

    d_list = np.arange(d_min, d_gap + d_max, d_gap)
    #k_list = np.array([1, 3, 10, 50, 100])
    k_list = np.array([1])

    outputs = []
    outputs_last = []
    for d in d_list:

        t = np.int(np.ceil(d/gamma))  # mark

        for k in k_list:

            regrets = defaultdict(MetricAggregator)
            cumregrets = defaultdict(MetricAggregator)
            thinnesses = defaultdict(MetricAggregator)
            errors = defaultdict(MetricAggregator)
            lambda_maxs = defaultdict(MetricAggregator)
            lambda_mins = defaultdict(MetricAggregator)
            log_maxs_over_mins = defaultdict(MetricAggregator)
            lambdas_second= defaultdict(MetricAggregator)
            lambdas_third = defaultdict(MetricAggregator)
            biases = defaultdict(MetricAggregator)
            variances = defaultdict(MetricAggregator)
            risks = defaultdict(MetricAggregator)

            for i in range(n):
                print(f"Running experiment {i} for dim {d} and k {k}...")
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
                    else:
                        if sim == 3:  # Run Russo Scenario for all methods
                            print('Run Russo Scenario for all methods')
                            results = russo_scenario_all(
                                d=d, k=k, t=t, state_factory=state_factory,
                                prior_var=prior_sd ** 2,
                                prior_mu=prior_mu,
                                noise_sd=noise_sd,
                                thin_thresh=thin_thresh,
                                const_infl=const_infl)
                        else:  # Run Russo Scenario
                            print('Run Russo Scenario 2')
                            results = russo_scenario2(
                                d=d, k=k, t=t, state_factory=state_factory,
                                prior_var=prior_sd ** 2,
                                prior_mu=prior_mu,
                                noise_sd=noise_sd,
                                thin_thresh=thin_thresh,
                                const_infl=const_infl)

                for name, (regret, thinness, error, lambda_max, lambda_min, lambda_second, lambda_third, bias, variance, risk) in results.items():
                    regrets[name].aggregate(regret)
                    cumregrets[name].aggregate(np.cumsum(regret))
                    # [name].aggregate(regret[-1])
                    thinnesses[name].aggregate(thinness)
                    errors[name].aggregate(error)
                    lambda_maxs[name].aggregate(lambda_max)
                    lambda_mins[name].aggregate(lambda_min)
                    log_maxs_over_mins[name].aggregate(
                        np.log(lambda_max)/lambda_min)
                    lambdas_second[name].aggregate(lambda_second)
                    lambdas_third[name].aggregate(lambda_third)
                    biases[name].aggregate(bias)
                    variances[name].aggregate(variance)
                    risks[name].aggregate(risk)

            metrics = {
                "regret": regrets,
                "cumregret": cumregrets,
                "thinnesses": thinnesses,
                "errors": errors,
                "lambda_maxs": lambda_maxs,
                "lambda_mins": lambda_mins,
                "log_maxs_over_mins": log_maxs_over_mins,
                "lambdas_second": lambdas_second, "lambdas_third":lambdas_third,
                "biases": biases,
                "variances": variances,
                "risks": risks,
            }

            labels = {"regret": "Instantaneous Regret", "cumregret": "Cumulative Regret", "thinnesses": "Thinness", "errors": "Normalized Error", "lambda_maxs": "Maximal Eigenvalues", "lambda_mins": "Minimal Eigenvalues", "log_maxs_over_mins": "Log Max / Min", "lambdas_second": "Second Largest Eigenvalues", "lambdas_third": "Third Largest Eigenvalues", "biases": "Biases", "variances": "Variances", "risks": "Risks"}

            output = pd.DataFrame()
            output_last = pd.DataFrame()

            for name, metric in metrics.items():
                # plt.clf()
                for alg, agg in metric.items():
                    #agg.plot(plt, alg)

                    mean, se = agg.get_mean_se()
                    nm = alg+'_'+name
                    output['d'] = d
                    output['k'] = k
                    output[nm+'_mean'] = mean
                    output[nm+'_se'] = se

                    output_last['d'] = d
                    output_last['k'] = k
                    output_last[nm+'_mean'] = [mean[-1]]
                    output_last[nm+'_se'] = [se[-1]]

            outputs.extend([output])
            outputs_last.extend([output_last])
    outputs = pd.concat(outputs)
    outputs_last = pd.concat(outputs_last)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    outputs.to_csv(f"outputs2/all-unit-{timestr}-{n}-{d_min}-{d_max}-{d_gap}-{k}-{t}-{sim}-{prior_mu}-{prior_sd}-{noise_sd}-{thin_thresh}-{const_infl}-{gamma}.csv", index=False)


    outputs_last.to_csv(f"outputs2/all-unit-last-{timestr}-{n}-{d_min}-{d_max}-{d_gap}-{k}-{t}-{sim}-{prior_mu}-{prior_sd}-{noise_sd}-{thin_thresh}-{const_infl}-{gamma}.csv", index=False)

    for name, metric in metrics.items():
        plt.clf()
        for alg, agg in metric.items():

            for d in d_list:
                nm = alg+'_'+name

                output_ = outputs_last.loc[outputs_last['d'] == d]

                mean = output_[nm+'_mean']
                se = output_[nm+'_se']

                x = output_['k']
                scale = 2.0
                lower = mean - scale * se
                upper = mean + scale * se

                plt.fill_between(x, lower, upper, alpha=0.2)
                plt.plot(x, mean, label=alg+'_dim_'+str(d))

        plt.xlabel("k")
        plt.ylabel(labels[name])

        plt.legend()
        plt.savefig(
            f"outputs2/{timestr}-unit-{'last_k'}-{name}-{n}-{d_min}-{d_max}-{d_gap}-{k}-{t}-{sim}-{prior_mu}-{prior_sd}-{noise_sd}-{thin_thresh}-{const_infl}-{gamma}.pdf")


def __main__():
    parser = argparse.ArgumentParser(description="Run simulations for various ROFUL algorithms.")

    parser.add_argument("-n", type=int, help="number of seeds", default=20)
    parser.add_argument("-k", type=int, help="number of actions", default=100)
    parser.add_argument("-d_min", type=int, help="dimension", default=50)
    parser.add_argument("-d_max", type=int, help="dimension", default=500)
    parser.add_argument("-d_gap", type=int, help="dimension", default=50)
    parser.add_argument("-t", type=int, help="time horizon", default=10000)
    parser.add_argument("-s", type=int, help="random seed", default=1)
    parser.add_argument("-pm", type=float, help="prior mu", default=0.0)
    parser.add_argument("-psd", type=float, help="prior standard deviation", default=1.0)

    parser.add_argument("-nsd", type=float, help="noise standard deviation", default=1.0)

    parser.add_argument("-th", type=float, help="threshold for thinness based inflation", default=2.0)

    parser.add_argument("-inf", type=float, help="inflation used when large thinness", default=5.0)

    parser.add_argument("-sim", type=int, help="0: russo scenario, 1: example 1, 2: example 2", default=0)

    parser.add_argument("-gamma", type=float, help="ratio", default=1)
    parser.add_argument("-mode", type=int, help="0: k, 1: prior_sd", default=0)
    args = parser.parse_args()
    '''

    '''
    if args.mode == 0:
        run_experiments_k(n=args.n, d_min=args.d_min, d_max=args.d_max, d_gap=args.d_gap, k=args.k, t=args.t, s=args.s, prior_mu=args.pm, prior_sd=args.psd, noise_sd=args.nsd, thin_thresh=args.th, const_infl=args.inf, sim=args.sim, gamma=args.gamma)

    elif args.mode == 1:
        run_experiments_prior_sd(n=args.n, d_min=args.d_min, d_max=args.d_max, d_gap=args.d_gap, k=args.k, t=args.t, s=args.s, prior_mu=args.pm, prior_sd=args.psd, noise_sd=args.nsd, thin_thresh=args.th, const_infl=args.inf, sim=args.sim, gamma=args.gamma)


if __name__ == "__main__":
    __main__()

# PYTHONPATH=src python -m experiments -sim 1 -k 3 -d 300 -t 1000 -pm 0 -psd 1 -nsd 10
# PYTHONPATH=src python -m experiments -sim 2 -k 3 -d 300 -t 1000 -pm 10 -psd 1 -nsd 1
# PYTHONPATH=src python -m experiments_2 -sim 0 -k 3 -d_min 5 -d_max 15 -d_gap 5 -t 1000 -pm 0 -psd 1 -nsd 5
# PYTHONPATH=src python -m experiments_2 -sim 0 -k 3 -d_min 5 -d_max 15 -d_gap 5 -t 1000 -pm 0 -psd 1 -nsd 5 -n 2
# PYTHONPATH=src python -m experiments_2 -sim 0 -k 3 -d_min 5 -d_max 50 -d_gap 5 -t 1000 -pm 0 -psd 1 -nsd 5 -n 10
# PYTHONPATH=src python -m experiments_2 -sim 0 -k 10 -d_min 50 -d_max 500 -d_gap 50 -t 1000 -pm 0 -psd 1 -nsd 5 -n 10
# PYTHONPATH=src python -m experiments_2 -sim 0 -k 10 -d_min 50 -d_max 500 -d_gap 50 -t 1000 -pm 0 -psd 1 -nsd 5 -n 10
# PYTHONPATH=src python -m experiments_2 -sim 0 -k 10 -d_min 50 -d_max 400 -d_gap 50 -t 1000 -pm 0 -psd 1 -nsd 5 -n 10 -gamma 1.2
# PYTHONPATH=src python -m experiments_2 -sim 0 -k 1 -d_min 50 -d_max 800 -d_gap 50 -t 1000 -pm 0 -psd 1 -nsd 5 -n 10 -gamma 0.8
# PYTHONPATH=src python -m experiments_2 -sim 0 -k 1 -d_min 50 -d_max 400 -d_gap 50 -t 1000 -pm 0 -psd 1 -nsd 1 -n 10 -gamma 0.8
# PYTHONPATH=src python -m experiments_2 -sim 0 -k 1 -d_min 50 -d_max 400 -d_gap 50 -t 1000 -pm 0 -psd 1 -nsd 0.2 -n 10 -gamma 0.8
# PYTHONPATH=src python -m experiments_2 -sim 0 -k 1 -d_min 50 -d_max 400 -d_gap 50 -t 1000 -pm 0 -psd 1 -nsd 0 -n 10 -gamma 0.8
# PYTHONPATH=src python -m experiments_2 -sim 3 -k 1 -d_min 50 -d_max 200 -d_gap 50 -t 1000 -pm 0 -psd 1 -nsd 0.2 -n 10 -gamma 0.8
# PYTHONPATH=src python -m experiments_2 -sim 3 -k 1 -d_min 50 -d_max 200 -d_gap 50 -t 1000 -pm 0 -psd 1 -nsd 0.2 -n 10 -gamma 0.8


# PYTHONPATH=src python -m experiments_2 -sim 0 -k 1 -d_min 50 -d_max 400 -d_gap 50 -t 1000 -pm 0 -psd 10 -nsd 1 -n 10 -gamma 0.8
# PYTHONPATH=src python -m experiments_2 -sim 0 -k 3 -d_min 50 -d_max 400 -d_gap 50 -t 1000 -pm 0 -psd 10 -nsd 1 -n 10 -gamma 0.8
# PYTHONPATH=src python -m experiments_2 -sim 0 -k 10 -d_min 50 -d_max 400 -d_gap 50 -t 1000 -pm 0 -psd 10 -nsd 1 -n 10 -gamma 0.8


# PYTHONPATH=src python -m experiments_2 -sim 0 -k 0 -d_min 50 -d_max 300 -d_gap 50 -t 1000 -pm 0 -psd 10 -nsd 1 -n 10 -gamma 0.8
# PYTHONPATH=src python -m experiments_2 -sim 3 -k 0 -d_min 50 -d_max 200 -d_gap 50 -t 1000 -pm 0 -psd 10 -nsd 1 -n 10 -gamma 0.8

# PYTHONPATH=src python -m experiments_2 -sim 0 -k 0 -d_min 50 -d_max 800 -d_gap 50 -t 1000 -pm 0 -psd 10 -nsd 1 -n 10 -gamma 0.8

# PYTHONPATH=src python -m experiments_2 -sim 0 -k 0 -d_min 3 -d_max 30 -d_gap 3 -t 1000 -pm 0 -psd 1000 -nsd 1 -n 10 -gamma 0.001 
# PYTHONPATH=src python -m experiments_2 -sim 0 -k 0 -d_min 50 -d_max 200 -d_gap 50 -t 1000 -pm 0 -psd 2 -nsd 1 -n 10 -gamma 0.1

# PYTHONPATH=src python -m experiments_2 -sim 0 -k 1 -d_min 50 -d_max 200 -d_gap 50 -t 1000 -pm 0 -psd 2 -nsd 1 -n 10 -gamma 0.01

# PYTHONPATH=src python -m experiments_2 -sim 0 -k 1 -d_min 50 -d_max 200 -d_gap 50 -t 1000 -pm 0 -psd 1000 -nsd 1 -n 10 -gamma 0.01 -mode 0

# PYTHONPATH=src python -m experiments_2 -sim 0 -k 0 -d_min 3 -d_max 30 -d_gap 3 -t 1000 -pm 0 -psd 1000 -nsd 1 -n 10 -gamma 0.001 -mode 0

# PYTHONPATH=src python -m experiments_2 -sim 0 -k 0 -d_min 3 -d_max 30 -d_gap 3 -t 1000 -pm 0 -psd 2 -nsd 1 -n 10 -gamma 0.001 -mode 0

# PYTHONPATH=src python -m experiments_2 -sim 0 -k 0 -d_min 3 -d_max 6 -d_gap 3 -t 1000 -pm 0 -psd 1000 -nsd 1 -n 10 -gamma 0.1 -mode 0