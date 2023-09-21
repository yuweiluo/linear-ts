import argparse
import abc

import numpy as np
import pandas as pd
import numpy.linalg as npl
import numpy.random as npr
import math

import torch

from envs import Context
from envs import Feedback
from utils import DataSummary


if torch.cuda.is_available():
    dev = "cuda:1"
else:
    dev = "cpu"
device = torch.device(dev)


class WorthFunction:
    summary: DataSummary

    def bind(self, summary: DataSummary):
        self.summary = summary
        self.update()

    def update(self):
        pass

    @abc.abstractmethod
    def compute(self, ctx: Context) -> np.ndarray:
        pass


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

    @property
    def d(self):
        return self.summary.d

    def choose_arm(self, ctx: Context) -> int:
        self.worth_func.bind(self.summary)  # mark
        values = self.worth_func.compute(ctx)

        return np.argmax(values).item()

    # @timing
    def update_model(self, feedback: Feedback):
        x = feedback.chosen_arm
        y = feedback.rew

        self.summary.add_obs(x, y)

    # @timing
    def update_metrics(self, feedback: Feedback):
        super().update_metrics(feedback)

        # with timing_block('appending'):
        lambda_max = max(self.summary.scale)
        lambda_min = min(self.summary.scale)

        self.beta = self.summary.radius_det()

        theta_hat = self.summary.mean

        # append the results to the list
        self.thinnesses.append(self.summary.thinness)
        self.lambda_max.append(lambda_max)
        self.lambda_min.append(lambda_min)

        self.worst_alpha.append(np.sqrt(lambda_min/lambda_max))

        self.proj_first.append(
            theta_hat.T @ self.summary.xx @ theta_hat/lambda_max/npl.norm(theta_hat)**2)

        beta_TS = self.inflation(self.summary) * self.summary.radius_det()

        assert ((self.beta+beta_TS)/self.beta == (self.summary.radius_TS() +
                self.summary.radius_det())/self.summary.radius_det())

        self.iota.append(self.inflation(self.summary))
        self.betas_TS.append(beta_TS)
        self.betas.append(self.beta)

        self.outputs = (self.worth_func.alphas, self.worth_func.mus, self.worst_alpha, self.metrics.regrets, self.thinnesses, self.iota,
                        self.lambda_max, self.lambda_min,  self.proj_first,  self.betas, self.betas_TS, self.worth_func.discrete_alphas)


class Feedback:
    ctx: Context
    arm_idx: int

    noise: float
    max_rew: float
    mean_rew: float

    def __init__(self, ctx, arm_idx, mean_rew, noise, max_rew, oracle_rew=None):
        self.ctx = ctx
        self.arm_idx = arm_idx

        self.noise = noise
        self.max_rew = max_rew
        self.mean_rew = mean_rew
        self.oracle_rew = oracle_rew

    @property
    def t(self):
        return self.ctx.t

    @property
    def arms(self):
        return self.ctx.arms

    @property
    def chosen_arm(self):
        return self.ctx.arms[self.arm_idx]

    @property
    def rew(self):
        return self.mean_rew + self.noise

    @property
    def regret(self):
        return self.max_rew - self.mean_rew

    @property
    def oracle_regret(self):
        return self.oracle_rew - self.mean_rew

    def __repr__(self):
        return f"LinFb(arm={self.arm_idx}, reg={self.regret}, noise={self.noise}, mean={self.mean_rew})"
