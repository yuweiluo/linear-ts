import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from collections import defaultdict
from utils import MetricAggregator

class save_results(results, save_path):
    def __init__(self):
        self.regrets = defaultdict(MetricAggregator)
        #self.o_regrets = defaultdict(MetricAggregator)
        self.cumregrets = defaultdict(MetricAggregator)
        #self.o_cumregrets = defaultdict(MetricAggregator)
        self.thinnesses = defaultdict(MetricAggregator)
        self.errors = defaultdict(MetricAggregator)
        #self.errors_candidate = defaultdict(MetricAggregator)
        #self.errors_pcandidate = defaultdict(MetricAggregator)
        self.lambda_maxs = defaultdict(MetricAggregator)
        self.lambda_mins = defaultdict(MetricAggregator)
        self.lambda_maxs_over_mins = defaultdict(MetricAggregator)
        self.worst_alphas = defaultdict(MetricAggregator)
        self.approx_alphas = defaultdict(MetricAggregator)
        self.discrete_alphas = defaultdict(MetricAggregator)
        self.mus = defaultdict(MetricAggregator)
        self.cumumus = defaultdict(MetricAggregator)
        self.betas = defaultdict(MetricAggregator)
        self.betas_TS = defaultdict(MetricAggregator)
        self.zetas = defaultdict(MetricAggregator)
        #self.log_maxs_over_mins = defaultdict(MetricAggregator)
        #self.lambdas_second= defaultdict(MetricAggregator)
        #self.lambdas_third = defaultdict(MetricAggregator)
        #self.lambdas_d_minus_1 = defaultdict(MetricAggregator)
        #self.lambdas_half_d = defaultdict(MetricAggregator)
        self.projs_first = defaultdict(MetricAggregator)

        #self.biases = defaultdict(MetricAggregator)
        #self.variances = defaultdict(MetricAggregator)
        #self.risks = defaultdict(MetricAggregator)
        
        self.labels = {"o_regret":"Oracle Instantaneous Regret",  
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

    def aggregate(self, results):
        for name, ( approx_alpha, mu, worst_alpha, regret, thinness, error, lambda_max, lambda_min, proj_first,beta,beta_TS,discrete_alpha) in results.items():
            #self.o_regrets[name].aggregate(o_rerget)
            #self.o_cumregrets[name].aggregate(np.cumsum(o_rerget))
            self.regrets[name].aggregate(regret)
            self.cumregrets[name].aggregate(np.cumsum(regret))
            self.thinnesses[name].aggregate(thinness)
            self.worst_alphas[name].aggregate(worst_alpha)
            self.approx_alphas[name].aggregate(approx_alpha)
            self.discrete_alphas[name].aggregate(discrete_alpha)
            self.mus[name].aggregate(mu)
            self.cumumus[name].aggregate(np.sqrt(np.cumsum(np.square(mu))))
            self.betas[name].aggregate(beta)
            self.betas_TS[name].aggregate(beta_TS)
            #self.zetas[name].aggregate(zeta)

            self.errors[name].aggregate(error)
            #self.errors_candidate[name].aggregate(error_candidate)
            #self.errors_pcandidate[name].aggregate(error_pcandidate)

            self.lambda_maxs[name].aggregate(lambda_max)
            self.lambda_mins[name].aggregate(lambda_min)
            #self.lambda_maxs_over_mins[name].aggregate(lambda_max_over_min)
            #self.log_maxs_over_mins[name].aggregate(
                #np.log(lambda_max)/lambda_min)
            #self.lambdas_second[name].aggregate(lambda_second)
            #self.lambdas_third[name].aggregate(lambda_third)
            #self.lambdas_d_minus_1[name].aggregate(lambda_d_minus_1)
            #self.lambdas_half_d[name].aggregate(lambda_half_d)
            #self.biases[name].aggregate(bias)
            #self.variances[name].aggregate(variance)
            #self.risks[name].aggregate(risk)
            self.projs_first[name].aggregate(proj_first)
            
    def get_metrics(self):
            self.metrics = {
                #'o_regret': o_regrets,
                #'o_cumregret': o_cumregrets,
                "regret": self.regrets,
                "cumregret": self.cumregrets,
                "thinnesses": self.thinnesses,
                "errors": self.errors,
                "lambda_maxs": self.lambda_maxs,
                "lambda_mins": self.lambda_mins,
                "lambda_maxs_over_mins": self.lambda_maxs_over_mins,
                #"log_maxs_over_mins": log_maxs_over_mins,
                #"lambdas_second": lambdas_second, 
                #"lambdas_third":lambdas_third,
                #"lambdas_d_minus_1": lambdas_d_minus_1,
                #"lambdas_half_d": lambdas_half_d,
                #"biases": biases,
                #"variances": variances,
                #"risks": risks,
                "projs_first": self.projs_first,
                #"errors_candidate": errors_candidate,
                #"errors_pcandidate": errors_pcandidate,
                "worst_alphas": self.worst_alphas,
                "approx_alphas": self.approx_alphas,
                #"oracle_alphas": oracle_alphas,
                "mus": self.mus,
                "cumumus": self.cumumus,
                "betas": self.betas,
                "zetas": self.zetas,
                "betas_TS": self.betas_TS,
                "discrete_alphas": self.discrete_alphas,
            }
        return self.metrics


