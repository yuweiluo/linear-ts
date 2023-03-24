import argparse

from collections import defaultdict
from ctypes import util

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from utils import MetricAggregator
from utils import StateFactory
from utils import predicted_risk

from plotting import *
from examples2 import *

import time
import sys
import os
if 'plotting' in sys.modules:  
    del sys.modules["plotting"]





def run_experiments_d(
    n, d_min, d_max, d_gap, k, t, s, prior_mu=0.0, prior_sd=10.0, noise_sd=1.0,
    thin_thresh=2.0,
    const_infl=5.0,
    sim=0,
    gamma=1,
    radius=1.0,
    alpha = 0.5
):
    state_factory = StateFactory(s)

    d_list = np.arange(np.int64(d_min), np.int64(d_gap + d_max), np.int64(d_gap))
    
    #k_list = np.array([1, 3, 10, 50, 100])
    #k_list = np.array([1,10,100])
    #k_list = np.array([0])
    k_list = np.array([1,3,10,100])
    #k_list = np.array([np.inf])

    outputs = []
    outputs_last = []

    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_name = f"dims-{d_min}-{d_max}-{d_gap}-gamma-{gamma}-radius-{radius}-prior_sd-{prior_sd}-noise_sd-{noise_sd}-sim-{sim}-k-{k}-t-{t}-prior_mu-{prior_mu}-thin_thresh-{thin_thresh}-const_infl-{const_infl}-n-{n}-time-{timestr}-alpha-{alpha}"

    output_folder_name = 'my_outputs'


    for d in d_list:

        t = np.int64(np.ceil(d/gamma))  # mark

        # predicted_risk_ = predicted_risk(gamma, radius, noise_sd)

        for k in k_list:

            regrets = defaultdict(MetricAggregator)
            o_regrets = defaultdict(MetricAggregator)
            cumregrets = defaultdict(MetricAggregator)
            o_cumregrets = defaultdict(MetricAggregator)
            thinnesses = defaultdict(MetricAggregator)
            errors = defaultdict(MetricAggregator)
            errors_candidate = defaultdict(MetricAggregator)
            errors_pcandidate = defaultdict(MetricAggregator)
            lambda_maxs = defaultdict(MetricAggregator)
            lambda_mins = defaultdict(MetricAggregator)
            lambda_maxs_over_mins = defaultdict(MetricAggregator)
            worst_alphas = defaultdict(MetricAggregator)
            approx_alphas = defaultdict(MetricAggregator)
            betas = defaultdict(MetricAggregator)
            zetas = defaultdict(MetricAggregator)
            log_maxs_over_mins = defaultdict(MetricAggregator)
            lambdas_second= defaultdict(MetricAggregator)
            lambdas_third = defaultdict(MetricAggregator)
            lambdas_d_minus_1 = defaultdict(MetricAggregator)
            lambdas_half_d = defaultdict(MetricAggregator)
            projs_first = defaultdict(MetricAggregator)

            biases = defaultdict(MetricAggregator)
            variances = defaultdict(MetricAggregator)
            risks = defaultdict(MetricAggregator)

            for i in range(n):
                print(f"Running experiment {i} for dim {d} and k {k}...")
                if sim == 1:  # Run Example 1
                    print('Run Example 1')
                    results = example1_scenario(
                        d=d, k=k, t=t, state_factory=state_factory,
                        prior_var=prior_sd **2 ,
                        prior_mu=prior_mu,
                        noise_sd=noise_sd,
                        thin_thresh=thin_thresh,
                        const_infl=const_infl,
                        radius = radius,
                        alpha = alpha)
                elif sim ==2:  
                    print('Run Example 2')
                    results = example2_scenario(
                        d=d, k=k, t=t, state_factory=state_factory,
                        prior_var=prior_sd **2 ,
                        noise_sd=noise_sd,
                        thin_thresh=thin_thresh,
                        const_infl=const_infl,
                        radius = radius,
                        alpha = alpha)
                elif sim ==3:  # non-grouped actions
                    print('run_single_experiment')
                    results = run_single_experiment(
                        d=50, k=10, t=t, l = 1, state_factory=state_factory,
                        prior_var=prior_sd **2 ,
                        noise_sd=noise_sd,
                        thin_thresh=thin_thresh,
                        const_infl=const_infl,
                        g= 0,
                        alpha = alpha)
                elif sim ==4:  # grouped actions
                    print('run_single_experiment_group')
                    results = run_single_experiment(
                        d=50, k=10, t=t, l = 1, state_factory=state_factory,
                        prior_var=prior_sd **2 ,
                        noise_sd=noise_sd,
                        thin_thresh=thin_thresh,
                        const_infl=const_infl,
                        g= 1,
                        alpha = alpha)
                elif sim ==0:  # Run Russo Scenario
                        print('Run Russo Scenario')
                        results = russo_scenario(
                            d=d, k=k, t=t, state_factory=state_factory,
                            prior_var=prior_sd ** 2,
                            prior_mu=prior_mu,
                            noise_sd=noise_sd,
                            thin_thresh=thin_thresh,
                            const_infl=const_infl,
                            radius = radius,
                            alpha = alpha)

                #for name, (regret, thinness, error, lambda_max, lambda_min, lambda_second, lambda_third, bias, variance, risk, lambda_d_minus_1, lambda_half_d, proj_first, error_candidate, error_pcandidate) in results.items():
                for name, (o_rerget, regret, thinness, error, lambda_max, lambda_min, lambda_max_over_min, proj_first, worst_alpha, approx_alpha, beta,zeta) in results.items():
                    o_regrets[name].aggregate(o_rerget)
                    o_cumregrets[name].aggregate(np.cumsum(o_rerget))
                    regrets[name].aggregate(regret)
                    cumregrets[name].aggregate(np.cumsum(regret))
                    # [name].aggregate(regret[-1])
                    thinnesses[name].aggregate(thinness)
                    worst_alphas[name].aggregate(worst_alpha)
                    approx_alphas[name].aggregate(approx_alpha)
                    betas[name].aggregate(beta)
                    zetas[name].aggregate(zeta)

                    errors[name].aggregate(error)
                    #errors_candidate[name].aggregate(error_candidate)
                    #errors_pcandidate[name].aggregate(error_pcandidate)

                    lambda_maxs[name].aggregate(lambda_max)
                    lambda_mins[name].aggregate(lambda_min)
                    lambda_maxs_over_mins[name].aggregate(lambda_max_over_min)
                    #log_maxs_over_mins[name].aggregate(
                        #np.log(lambda_max)/lambda_min)
                    #lambdas_second[name].aggregate(lambda_second)
                    #lambdas_third[name].aggregate(lambda_third)
                    #lambdas_d_minus_1[name].aggregate(lambda_d_minus_1)
                    #lambdas_half_d[name].aggregate(lambda_half_d)
                    # print(f"bias: {bias}, {type(bias)}, d: {d}, {type(d)}")
                    #biases[name].aggregate(bias)
                    #variances[name].aggregate(variance)
                    #risks[name].aggregate(risk)
                    projs_first[name].aggregate(proj_first)

            metrics = {
                'o_regret': o_regrets,
                'o_cumregret': o_cumregrets,
                "regret": regrets,
                "cumregret": cumregrets,
                "thinnesses": thinnesses,
                "errors": errors,
                "lambda_maxs": lambda_maxs,
                "lambda_mins": lambda_mins,
                "lambda_maxs_over_mins": lambda_maxs_over_mins,
                #"log_maxs_over_mins": log_maxs_over_mins,
                #"lambdas_second": lambdas_second, 
                #"lambdas_third":lambdas_third,
                #"lambdas_d_minus_1": lambdas_d_minus_1,
                #"lambdas_half_d": lambdas_half_d,
                #"biases": biases,
                #"variances": variances,
                #"risks": risks,
                "projs_first": projs_first,
                #"errors_candidate": errors_candidate,
                #"errors_pcandidate": errors_pcandidate,
                "worst_alphas": worst_alphas,
                "approx_alphas": approx_alphas,
                "betas": betas,
                "zetas": zetas
            }

            labels = {"o_regret":"Oracle Instantaneous Regret",  "o_cumregret": "Oracle Cumulative Regret","regret": "Instantaneous Regret", "cumregret": "Cumulative Regret", "thinnesses": "Thinness", "errors": "Normalized Error", "lambda_maxs": "Maximal Eigenvalues", "lambda_mins": "Minimal Eigenvalues", "log_maxs_over_mins": "Log Max / Min", "lambdas_second": "Second Largest Eigenvalues", "lambdas_third": "Third Largest Eigenvalues", "biases": "Biases", "variances": "Variances", "risks": "Risks", "lambdas_d_minus_1": "d-1 Largest Eigenvalues", "lambdas_half_d": "d/2 Largest Eigenvalues", "projs_first": "$\\frac{\|\|\hat{\Theta}_{t-1}\|\|_{V_{t-1}}^2/\|\|\hat{\Theta}_{t-1}\|\|^2}{\lambda_{\max}(V_{t-1})} $", "errors_candidate": "Normalized Error of Candidate", "errors_pcandidate": "Normalized Error of P-Candidate", "worst_alphas": "Worst Alpha", "betas": "Beta", "zetas": "Zeta", "lambda_maxs_over_mins": "Max / Min", "approx_alphas": "$\hat{\\alpha}_t$"}



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

            isExist = os.path.exists(output_folder_name)
            if not isExist:
                os.makedirs(output_folder_name)
            figure_folder_name = f"figures/figures-{output_name}"
            isExist = os.path.exists(figure_folder_name)
            if not isExist:
                os.makedirs(figure_folder_name)

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
                plt.savefig(f"{figure_folder_name}/k = {k}-{name}-{n}-{d}-{k}-{t}-{sim}-{prior_mu}-{prior_sd}-{noise_sd}-{thin_thresh}-{const_infl}.jpg", dpi = 600)
                #output.to_csv(f"{figure_folder_name}/k = {k}{name}-{n}-{d}-{k}-{t}-{sim}-{prior_mu}-{prior_sd}-{noise_sd}-{thin_thresh}-{const_infl}.csv", index=False)

    outputs = pd.concat(outputs)
    outputs_last = pd.concat(outputs_last)


    
    outputs.to_csv(f"{output_folder_name}/all-{output_name}.csv", index=False)
    outputs_last.to_csv(f"{output_folder_name}/all-last-{output_name}.csv", index=False)



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
                plt.xscale('log')


            if name == 'errors':
                plt.axhline(y=predicted_risk_[2], linestyle= '--', color='red', label='Predicted Risk')
        plt.xlabel("k")
        plt.ylabel(labels[name])

        plt.legend()
        plt.savefig(
            f"{figure_folder_name}/{'last_k'}-{name}-{output_name}.jpg", dpi = 600)


            

    plot_statistics(output_folder_name, output_name, figure_folder_name, mode = 'd', gamma = gamma)
    outputs.to_csv(f"{figure_folder_name}/all-{output_name}.csv", index=False)
    outputs_last.to_csv(f"{figure_folder_name}/all-last-{output_name}.csv", index=False)





def __main__():
    parser = argparse.ArgumentParser(description="Run simulations for various ROFUL algorithms.")

    parser.add_argument("-n", type=np.int64, help="number of seeds", default=20)
    parser.add_argument("-k", type=np.int64, help="number of actions", default=100)
    parser.add_argument("-para_min", type=np.float64, help="parameter to simulate", default=50)
    parser.add_argument("-para_max", type=np.float64, help="parameter to simulate", default=500)
    parser.add_argument("-para_gap", type=np.float64, help="parameter to simulate", default=50)
    parser.add_argument("-t", type=np.int64, help="time horizon", default=200)
    parser.add_argument("-s", type=np.int64, help="random seed", default=1)
    parser.add_argument("-pm", type=np.float64, help="prior mu", default=0.0)
    parser.add_argument("-psd", type=np.float64, help="prior standard deviation", default=1.0)

    parser.add_argument("-nsd", type=np.float64, help="noise standard deviation", default=1.0)

    parser.add_argument("-th", type=np.float64, help="threshold for thinness based inflation", default=2.0)

    parser.add_argument("-inf", type=np.float64, help="inflation used when large thinness", default=5.0)

    parser.add_argument("-sim", type=np.int64, help="0: russo scenario, 1: example 1, 2: example 2", default=0)

    parser.add_argument("-gamma", type=np.float64, help="ratio", default=0.1)
    parser.add_argument("-mode", type=str, help="mode", default="d")
    parser.add_argument("-radius", type=np.float64, help="norm of beta", default=1.0)
    parser.add_argument("-alpha", type=np.float64, help="ratio threshold", default=0.5)
    args = parser.parse_args()
    '''

    '''
    if args.mode == 'd':
        run_experiments_d(n=args.n, d_min=args.para_min, d_max=args.para_max, d_gap=args.para_gap, k=args.k, t=args.t, s=args.s, prior_mu=args.pm, prior_sd=args.psd, noise_sd=args.nsd, thin_thresh=args.th, const_infl=args.inf, sim=args.sim, gamma=args.gamma, radius = args.radius, alpha = args.alpha)



if __name__ == "__main__":
    __main__()



# PYTHONPATH=src python -m experiments_2 -sim 2 -k 0 -d_min 30 -d_max 60 -d_gap 30 -t 1000 -pm 0 -psd 1000 -nsd 1 -n 10 -gamma 0.1 -mode 0

# PYTHONPATH=src python -m experiments_2 -sim 2 -k 0 -para_min 0.1 -para_max 10 -para_gap 0.3 -t 1000 -pm 0 -psd 1000 -nsd 1 -n 10 -mode "gamma" -radius 5

# PYTHONPATH=src python -m experiments_2 -sim 2 -k 0 -para_min 50 -para_max 100 -para_gap 50 -pm 0 -nsd 1 -n 10 -mode "d" -gamma 0.1 -psd 1000 -radius 1
# PYTHONPATH=src python -m experiments_2 -sim 2 -k 0 -para_min 5 -para_max 20 -para_gap 5 -pm 0 -nsd 1 -n 10 -mode "d" -gamma 0.01 -psd 1000 -radius 1
# PYTHONPATH=src python -m experiments_2 -sim 2 -k 0 -para_min 50 -para_max 200 -para_gap 50 -pm 0 -nsd 1 -n 10 -mode "d" -gamma 10 -psd 1000 -radius 1

# PYTHONPATH=src python -m experiments_2 -sim 2 -k 0 -para_min 50 -para_max 200 -para_gap 50 -pm 0 -nsd 1 -n 10 -mode "d" -gamma 0.5 -psd 1000 -radius 1 -s 2
# 
# PYTHONPATH=src python -m experiments_2 -sim 2 -k 0 -para_min 5 -para_max 20 -para_gap 5 -pm 0 -nsd 1 -n 10 -mode "d" -gamma 0.5 -psd 1000 -radius 1 -s 2  

## Mar 9 2023
##  PYTHONPATH=src python -m experiments2 -sim 4 -k 0 -para_min 30 -para_max 30 -para_gap 10 -pm 0 -nsd 1 -n 30 -mode "d" -gamma 0.1 -psd 10 -radius 1 -s 1
##  PYTHONPATH=src python -m experiments2 -sim 0 -k 100 -para_min 10 -para_max 10 -para_gap 10 -pm 0 -nsd 1 -n 20 -mode "d" -gamma 0.002 -psd 10 -radius 10 -s 1