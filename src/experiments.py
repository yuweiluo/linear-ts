import argparse

from collections import defaultdict
from ctypes import util

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from utils import MetricAggregator
from utils import StateFactory
from utils import predicted_risk
from utils import save_results

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
    alpha = 0.5
):
    state_factory = StateFactory(s)

    d_list = np.arange(np.int64(d_min), np.int64(d_gap + d_max), np.int64(d_gap))
    
    #k_list = np.array([1, 3, 10, 50, 100])
    #k_list = np.array([1,10,100])
    #k_list = np.array([0])
    k_list = np.array([100])
    #k_list = np.array([np.inf])
    saver = save_results()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_name = f"dims-{d_min}-{d_max}-{d_gap}-gamma-{gamma}-radius-{radius}-prior_sd-{prior_sd}-noise_sd-{noise_sd}-sim-{sim}-k-{k}-t-{t}-prior_mu-{prior_mu}-thin_thresh-{thin_thresh}-const_infl-{const_infl}-n-{n}-time-{timestr}-alpha-{alpha}"

    output_folder_name = 'my_outputs'


    for d in d_list:

        t = np.int64(np.ceil(d/gamma))  # FIXME

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

            metrics, output, output_last = saver.aggregate_output()
            outputs, outputs_last = saver.save_outputs(output_folder_name,output_name)
            
            figure_folder_name = f"figures/figures-{output_name}"
            figure_name_suffix = f"-{k}-{n}-{d}-{k}-{t}-{sim}-{prior_mu}-{prior_sd}-{noise_sd}-{thin_thresh}-{const_infl}.jpg"
            plot_output(metrics, output, saver.labels, figure_folder_name,figure_name_suffix)

            
    # plot curves of each k together 
    # plot_statistics(output_folder_name, output_name, figure_folder_name, mode = 'd', gamma = gamma)

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








## Jun 23
# PYTHONPATH=src python -m experiments -sim 0 -k 0 -para_min 50 -para_max 50 -para_gap 20 -pm 0 -nsd 1 -n 20 -mode "d" -gamma 0.02 -psd 10 -radius 10 -s 1 -alpha 8.0

# PYTHONPATH=src python -m experiments -sim 4 -k 0 -para_min 50 -para_max 50 -para_gap 20 -pm 0 -nsd 1 -n 20 -mode "d" -gamma 0.02 -psd 10 -radius 10 -s 1 -alpha 12.0

# PYTHONPATH=src python -m experiments -sim 2 -k 3 -para_min 50 -para_max 50 -para_gap 20 -pm 10 -nsd 1 -n 20 -mode "d" -gamma 0.1 -psd 1 -radius 1 -s 1 -alpha 8.0

# PYTHONPATH=src python -m experiments -sim 5 -k 0 -para_min 50 -para_max 50 -para_gap 20 -pm 0 -nsd 1 -n 20 -mode "d" -gamma 0.02 -psd 10 -radius 10 -s 1 -alpha 8.0

# PYTHONPATH=src python -m experiments -sim 0 -k 0 -para_min 50 -para_max 50 -para_gap 20 -pm 0 -nsd 1 -n 20 -mode "d" -gamma 0.02 -psd 10 -radius 10 -s 1 -alpha 12.0

# PYTHONPATH=src python -m experiments -sim 0 -k 0 -para_min 5 -para_max 5 -para_gap 5 -pm 0 -nsd 1 -n 3 -mode "d" -gamma 0.02 -psd 10 -radius 10 -s 1 -alpha 12.0