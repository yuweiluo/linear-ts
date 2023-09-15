import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from collections import defaultdict
from utils import MetricAggregator

class save_results(results, save_path):
    def __init__(self):
        regrets = defaultdict(MetricAggregator)
        #o_regrets = defaultdict(MetricAggregator)
        cumregrets = defaultdict(MetricAggregator)
        #o_cumregrets = defaultdict(MetricAggregator)
        thinnesses = defaultdict(MetricAggregator)
        errors = defaultdict(MetricAggregator)
        #errors_candidate = defaultdict(MetricAggregator)
        #errors_pcandidate = defaultdict(MetricAggregator)
        lambda_maxs = defaultdict(MetricAggregator)
        lambda_mins = defaultdict(MetricAggregator)
        lambda_maxs_over_mins = defaultdict(MetricAggregator)
        worst_alphas = defaultdict(MetricAggregator)
        approx_alphas = defaultdict(MetricAggregator)
        discrete_alphas = defaultdict(MetricAggregator)
        mus = defaultdict(MetricAggregator)
        cumumus = defaultdict(MetricAggregator)
        betas = defaultdict(MetricAggregator)
        betas_TS = defaultdict(MetricAggregator)
        zetas = defaultdict(MetricAggregator)
        #log_maxs_over_mins = defaultdict(MetricAggregator)
        #lambdas_second= defaultdict(MetricAggregator)
        #lambdas_third = defaultdict(MetricAggregator)
        #lambdas_d_minus_1 = defaultdict(MetricAggregator)
        #lambdas_half_d = defaultdict(MetricAggregator)
        projs_first = defaultdict(MetricAggregator)

        #biases = defaultdict(MetricAggregator)
        #variances = defaultdict(MetricAggregator)
        #risks = defaultdict(MetricAggregator)


    for name, ( approx_alpha, mu, worst_alpha, regret, thinness, error, lambda_max, lambda_min, proj_first,beta,beta_TS,discrete_alpha) in results.items():
                        #o_regrets[name].aggregate(o_rerget)
                        #o_cumregrets[name].aggregate(np.cumsum(o_rerget))
                        regrets[name].aggregate(regret)
                        cumregrets[name].aggregate(np.cumsum(regret))
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
                        #biases[name].aggregate(bias)
                        #variances[name].aggregate(variance)
                        #risks[name].aggregate(risk)
                        projs_first[name].aggregate(proj_first)

    metrics = {
        #'o_regret': o_regrets,
        #'o_cumregret': o_cumregrets,
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
        #"oracle_alphas": oracle_alphas,
        "mus": mus,
        "cumumus": cumumus,
        "betas": betas,
        "zetas": zetas,
        "betas_TS": betas_TS,
        "discrete_alphas": discrete_alphas,
    }

    labels = {"o_regret":"Oracle Instantaneous Regret",  
        "o_cumregret": "Oracle Cumulative Regret",
        "regret": "Instantaneous Regret", 
        "cumregret": "Cumulative Regret", 
        "thinnesses": "Thinness", 
        "errors": "Normalized Error", 
        "lambda_maxs": "Maximal Eigenvalues", 
        "lambda_mins": "Minimal Eigenvalues", 
        "log_maxs_over_mins": "Log Max / Min", 
        "lambdas_second": "Second Largest Eigenvalues", 
        "lambdas_third": "Third Largest Eigenvalues", 
        "biases": "Biases", "variances": "Variances", 
        "risks": "Risks", 
        "lambdas_d_minus_1": "d-1 Largest Eigenvalues", 
        "lambdas_half_d": "d/2 Largest Eigenvalues", 
        "projs_first": "$\zeta_t$", 
        "errors_candidate": "Normalized Error of Candidate", 
        "errors_pcandidate": "Normalized Error of P-Candidate", 
        "worst_alphas": "Worst Alpha", 
        "betas": "Beta", 
        "zetas": "Zeta", 
        "lambda_maxs_over_mins": "Max / Min", 
        "approx_alphas": "$\hat{\\alpha}_t$", 
        "oracle_alphas": "Oracle Alphas", 
        "betas_TS": "Beta TS", 
        "mus": "$\mu_t$", 
        "cumumus": "$\\left(\sum_{t=1}^T\mu_t^2 \\right )^{1/2}$", 
        "discrete_alphas": "$\hat{\\alpha}_t$"
        }