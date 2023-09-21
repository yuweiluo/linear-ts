import argparse


import numpy as np
import pandas as pd

import torch


if torch.cuda.is_available():
    dev = "cuda:1"
else:
    dev = "cpu"
device = torch.device(dev)


class NeuralRoful:
    def __init__(self, d, lambda_, worth_func: WorthFunction,  param=None, noise_sd=None, t=None, radius=None, delta=None, inflation=None):
        super().__init__()

        self.t = t
        self.radius = radius
        self.delta = delta
        self.inflation = inflation

        self.tt = 0

        self.summary = DataSummary(
            d, lambda_, noise_sd, radius, param, inflation, self.t)
        self.worth_func = worth_func

        self.param = param
        self.noise_sd = noise_sd

        self.func0 = lambda x, c_0, gamma:  1/(1 + c_0 * gamma * x)
        self.func1 = lambda x, c_0, gamma:  x/(1 + c_0 * gamma * x)**2
        self.func2 = lambda x, c_0, gamma:  x**2/(1 + c_0 * gamma * x)**2

        # metrics
        self.approx_alpha = []
        self.betas = []
        self.betas_TS = []
        self.iota = []

        self.thinnesses = []
        self.oracle_regrets = []
        self.errors = []
        self.errors_candidate = []
        self.errors_pcandidate = []
        self.lambda_max = []
        self.lambda_min = []
        self.lambda_max_over_min = []

        self.worst_alpha = []
        self.zeta = []

        self.oracle_alpha = []
        self.lambda_second = []
        self.lambda_third = []
        self.lambda_d_minus_1 = []
        self.lambda_half_d = []
        self.proj_first = []
        self.B_n = []
        self.V_n = []
        self.R_n = []

    def choose_arm(self, ctx: Context) -> int:
        self.worth_func.bind(self.summary)  # mark
        values = self.worth_func.compute(ctx)

        return np.argmax(values).item()
