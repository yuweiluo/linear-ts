import numpy as np

from envs import Environment
from envs import NoiseGenerator
from envs import StochasticContextGenerator as CtxGenerator
from envs import GroupedStochasticContextGenerator as GroupedCtxGenerator
from envs import Example2ContextGenerator as Ex2CtxGenerator
from envs import Example1ContextGenerator as Ex1CtxGenerator
from envs import Context

from policies import Roful
from utils import timing_block



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
from examples import *

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
    alpha = 0.5,
    mu_TS = 10.0,
    mu_Greedy = 12.0,
    dataset = 'eeg'
):
    state_factory = StateFactory(s)

    d_list = np.arange(np.int64(d_min), np.int64(d_gap + d_max), np.int64(d_gap))
    
    #k_list = np.array([1, 3, 10, 50, 100])
    #k_list = np.array([1,10,100])
    #k_list = np.array([0])
    k_list = np.array([2])
    #k_list = np.array([np.inf])

    saver = save_results()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_name = f"dims-{d_min}-{d_max}-{d_gap}-gamma-{gamma}-radius-{radius}-prior_sd-{prior_sd}-noise_sd-{noise_sd}-sim-{sim}-k-{k}-t-{t}-prior_mu-{prior_mu}-thin_thresh-{thin_thresh}-const_infl-{const_infl}-n-{n}-time-{timestr}-mu_TS-{mu_TS} -mu_Greedy-{mu_Greedy}-dataset-{dataset}"

    output_folder_name = 'my_outputs'


    for d in d_list:

        t = np.int64(np.ceil(d/gamma))  # mark

        predicted_risk_ = predicted_risk(gamma, radius, noise_sd)

        for k in k_list:

            saver.init_outputs(d = d, k = k)

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
                elif sim ==5:  # classification problems
                    
                    print('run_classification_problems')
                    results = emp_scenario(
                        d=d, k=2, t=t, state_factory=state_factory,
                        prior_var=prior_sd **2 ,
                        noise_sd=noise_sd,
                        thin_thresh=thin_thresh,
                        const_infl=const_infl,
                        radius = radius,
                        alpha = alpha,
                        mu_TS = mu_TS,
                        mu_Greedy = mu_Greedy,
                        dataset = dataset)
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
                        
                saver.aggregate_metrics(results)

                #for name, (regret, thinness, error, lambda_max, lambda_min, lambda_second, lambda_third, bias, variance, risk, lambda_d_minus_1, lambda_half_d, proj_first, error_candidate, error_pcandidate) in results.items():
                for name, ( approx_alpha, mu, worst_alpha, regret, thinness, error, lambda_max, lambda_min, proj_first,beta,beta_TS,discrete_alpha) in results.items():
                    #o_regrets[name].aggregate(o_rerget)
                    #o_cumuregrets[name].aggregate(np.cumsum(o_rerget))
                    regrets[name].aggregate(regret)
                    cumuregrets[name].aggregate(np.cumsum(regret))
                    incorrect_frac[name].aggregate(np.cumsum(regret)/np.arange(1, t + 1))
                    # [name].aggregate(regret[-1])
                    thinnesses[name].aggregate(thinness)
                    worst_alphas[name].aggregate(worst_alpha)
                    approx_alphas[name].aggregate(approx_alpha)
                    discrete_alphas[name].aggregate(discrete_alpha)
                    mus[name].aggregate(mu)
                    cumumus[name].aggregate(np.sqrt(np.cumsum(np.square(mu))))
                    betas[name].aggregate(beta)
                    betas_TS[name].aggregate(beta_TS)
                    #zetas[name].aggregate(zeta)

                    errors[name].aggregate(error)
                    #errors_candidate[name].aggregate(error_candidate)
                    #errors_pcandidate[name].aggregate(error_pcandidate)

                    lambda_maxs[name].aggregate(lambda_max)
                    lambda_mins[name].aggregate(lambda_min)
                    #lambda_maxs_over_mins[name].aggregate(lambda_max_over_min)
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
                #'o_regret': o_regrets,
                #'o_cumuregret': o_cumuregrets,
                "regret": regrets,
                "cumuregret": cumuregrets,
                "incorrect_frac":incorrect_frac,
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
                #"oracle_alphas": oracle_alphas,
                "mus": mus,
                "cumumus": cumumus,
                "betas": betas,
                "zetas": zetas,
                "betas_TS": betas_TS,
                "discrete_alphas": discrete_alphas,
            }

            labels = {"o_regret":"Oracle Instantaneous Regret",  "o_cumuregret": "Oracle Cumulative Regret","regret": "Instantaneous Regret", "cumuregret": "Cumulative Regret", "thinnesses": "Thinness", "errors": "Normalized Error", "lambda_maxs": "Maximal Eigenvalues", "lambda_mins": "Minimal Eigenvalues", "log_maxs_over_mins": "Log Max / Min", "lambdas_second": "Second Largest Eigenvalues", "lambdas_third": "Third Largest Eigenvalues", "biases": "Biases", "variances": "Variances", "risks": "Risks", "lambdas_d_minus_1": "d-1 Largest Eigenvalues", "lambdas_half_d": "d/2 Largest Eigenvalues", "projs_first": "$\\frac{\|\|\hat{\Theta}_{t-1}\|\|_{V_{t-1}}^2/\|\|\hat{\Theta}_{t-1}\|\|^2}{\lambda_{\max}(V_{t-1})} $", "errors_candidate": "Normalized Error of Candidate", "errors_pcandidate": "Normalized Error of P-Candidate", "worst_alphas": "Worst Alpha", "betas": "Beta", "zetas": "Zeta", "lambda_maxs_over_mins": "Max / Min", "approx_alphas": "$\hat{\\alpha}_t$", "oracle_alphas": "Oracle Alphas", "betas_TS": "Beta TS", "mus": "$\mu_t$", "cumumus": "$\\left(\sum_{t=1}^T\mu_t^2 \\right )^{1/2}$", "discrete_alphas": "$\hat{\\alpha}_t$", "incorrect_frac": "Fraction of incorrect decisions"}


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


            os.makedirs(output_folder_name, exist_ok=True)
            figure_folder_name = f"figures/figures-{output_name}"
            os.makedirs(figure_folder_name, exist_ok=True)
            output = pd.DataFrame()
            
            
            markers = [6,4,5,7,'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']
            
            for name, metric in metrics.items():
                plt.clf()
                marker_index = 0
                max_y = []
                for alg, agg in metric.items():
                    agg.plot(plt, alg, marker=markers[marker_index], mark_num=10)

                    mean, se = agg.get_mean_se()
                    nm = alg+'_'+name
                    output[nm+'_mean'] = mean
                    output[nm+'_se'] = se
                    marker_index += 1
                    max_y.append(np.max(mean+se))
                # get the max mean for the for loop above to set the ylim:
                if name == "incorrect_frac":
                    if dataset == 'eeg':
                        plt.ylim(0.34, 0.54)
                    elif dataset == 'cardiotocography':
                        plt.ylim(0.0, 1.0 )
                    elif dataset == 'eye_movements':
                        plt.ylim(0.4, 0.8)
                if name == "cumuregret":
                    print(max_y)
                if name == "mus" or name == "discrete_alphas":
                    plt.yscale("log")
                plt.xlabel("Time")
                plt.ylabel(labels[name])

                plt.legend()
                plt.savefig(f"{figure_folder_name}/k = {k}-{name}-{n}-{d}-{k}-{t}-{sim}-{prior_mu}-{prior_sd}-{noise_sd}-{thin_thresh}-{const_infl}.jpg", dpi = 600)
                #output.to_csv(f"{figure_folder_name}/k = {k}{name}-{n}-{d}-{k}-{t}-{sim}-{prior_mu}-{prior_sd}-{noise_sd}-{thin_thresh}-{const_infl}.csv", index=False)

    outputs = pd.concat(outputs)
    outputs_last = pd.concat(outputs_last)


    
    outputs.to_csv(f"{output_folder_name}/all-{output_name}.csv", index=False)
    outputs_last.to_csv(f"{output_folder_name}/all-last-{output_name}.csv", index=False)


# plot statistics for multiple d in one figure
    markers = [6,4,5,7,'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']

    for name, metric in metrics.items():
        marker_index = 0
        plt.clf()
        for alg, agg in metric.items():

            for k in k_list:
                nm = alg+'_'+name

                output_ = outputs_last.loc[outputs_last['k'] == k]

                mean = output_[nm+'_mean']
                se = output_[nm+'_se']

                x = output_['d']
                scale = 2.0
                lower = mean - scale * se
                upper = mean + scale * se

                plt.fill_between(x, lower, upper, alpha=0.2)
                #plt.plot(x, mean, label=alg+'_k_'+str(k))
                plt.plot(x, mean, label=alg, marker=markers[marker_index])
                marker_index += 1
                #plt.plot(x, 0.5*x, label=alg+'0.5'+str(k))
                plt.xscale('log')
                #if name == 'mus' or name == 'cumumus' or name == 'approx_alphas' or name == 'discrete_alphas' or name == 'beta_TS':
                plt.yscale('log')


            if name == 'errors':
                plt.axhline(y=predicted_risk_[2], linestyle= '--', color='red', label='Predicted Risk')
        plt.xlabel("d")
        plt.ylabel(labels[name])

        plt.legend()
        plt.savefig(
            f"{figure_folder_name}/{'last_d'}-{name}-{output_name}.jpg", dpi = 600)


            
    #plot curves of each k together 
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
    parser.add_argument("-mu_TS", type=np.float64, help="ratio threshold for TS", default=10.0)
    parser.add_argument("-mu_Greedy", type=np.float64, help="ratio threshold for Greedy", default=12.0)
    parser.add_argument("-alpha", type=np.float64, help="placeholder", default=0.0)
    parser.add_argument("-dataset", type=str, help="dataset name", default="EEG")
    args = parser.parse_args()
    '''

    '''
    if args.mode == 'd':
        run_experiments_d(n=args.n, d_min=args.para_min, d_max=args.para_max, d_gap=args.para_gap, k=args.k, t=args.t, s=args.s, prior_mu=args.pm, prior_sd=args.psd, noise_sd=args.nsd, thin_thresh=args.th, const_infl=args.inf, sim=args.sim, gamma=args.gamma, radius = args.radius, alpha = args.alpha, mu_TS =args.mu_TS, mu_Greedy =args.mu_Greedy, dataset = args.dataset)



if __name__ == "__main__":
    __main__()










# PYTHONPATH=src python -m emp1 -sim 5 -k 0 -para_min 150 -para_max 150 -para_gap 20 -pm 0 -nsd 0.2 -n 1 -mode "d" -gamma 0.1 -psd 0.001 -radius 5 -s 1 -alpha 0.0 -dataset 'cardiotocography'

# PYTHONPATH=src python -m emp1 -sim 5 -k 0 -para_min 1450 -para_max 1450 -para_gap 20 -pm 0 -nsd 0.2 -n 20 -mode "d" -gamma 0.1 -psd 0.001 -radius 5 -s 1 -alpha 0.0 -dataset 'eeg'

# PYTHONPATH=src python -m emp1 -sim 5 -k 0 -para_min 1000 -para_max 1000 -para_gap 20 -pm 0 -nsd 0.1 -n 1 -mode "d" -gamma 0.1 -psd 0.001 -radius 5 -s 1 -alpha 8.0 -dataset 'eye_movements' 

# PYTHONPATH=src python -m emp1 -sim 5 -k 0 -para_min 1450 -para_max 1450 -para_gap 20 -pm 0 -nsd 0.2 -n 1 -mode "d" -gamma 0.1 -psd 0.01 -radius 5 -s 1 -alpha 0.0 -dataset 'satimage'