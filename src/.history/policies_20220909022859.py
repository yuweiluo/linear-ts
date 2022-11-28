import abc

from typing import List, Callable

import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt


from envs import Context
from envs import Feedback
from utils import DataSummary

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import fsolve


class Metric:
    regrets: List[float]

    def __init__(self):
        self.regrets = []

    def add_obs(self, feedback: Feedback):
        self.regrets.append(feedback.regret)


class Policy:
    metrics: Metric

    def __init__(self):
        self.metrics = Metric()

    @abc.abstractmethod
    def choose_arm(self, ctx: Context) -> int:
        pass

    def update(self, feedback: Feedback):
        self.update_model(feedback)
        self.update_metrics(feedback)

    @abc.abstractmethod
    def update_model(self, feedback: Feedback):
        pass

    def update_metrics(self, feedback: Feedback):
        self.metrics.add_obs(feedback)


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


class ProductWorthFunction(WorthFunction):
    def compute(self, ctx: Context) -> np.ndarray:
        values = ctx.arms @ self.candidates()

        if values.ndim == 2:
            values = np.max(values, axis=1)

        return values

    @abc.abstractmethod
    def candidates(self):
        pass


class Roful(Policy):
    thinnesses: List[float]
    # Yuwei

    summary: DataSummary
    worth_func: WorthFunction

    def __init__(self, d, prior_var, worth_func: WorthFunction,  param=None, noise_sd = None):
        super().__init__()

        self.t = 0
        self.thinnesses = []
        self.errors = []
        self.summary = DataSummary(d, prior_var)
        self.worth_func = worth_func
        self.all_rew = 0

        self.param = param
        self.lambda_max = []
        self.lambda_min = []
        self.lambda_second = []
        self.lambda_third = []
        self.B_n = []
        self.V_n = []
        self.R_n = []



        self.func0 = lambda x, c_0, gamma:  1/(1+ c_0 * gamma * x)
        self.func1 = lambda x, c_0, gamma:  x/(1+ c_0 * gamma * x)**2
        self.func2 = lambda x, c_0, gamma:  x**2/(1+ c_0 * gamma * x)**2


    @staticmethod
    # added param as input to calculate error
    def ts(d, prior_var, inflation=1.0, state=None, param=None, noise_sd = None):
        if isinstance(inflation, float):
            inflation = Roful.const_inflation(inflation)

        return Roful(d, prior_var, TsWorthFunction(inflation, state=state), param, noise_sd = noise_sd)

    def pure_thin(d, prior_var, inflation=1.0, state=None):
        if isinstance(inflation, float):
            inflation = Roful.const_inflation(inflation)

        return Roful(d, prior_var, PureThinnessWorthFunction(inflation, state=state))

    def thin_dirt(d, prior_var, inflation=1.0, state=None, param=None):
        if isinstance(inflation, float):
            inflation = Roful.const_inflation(inflation)

        return Roful(d, prior_var, ThinnessDirectedWorthFunction(inflation, state=state), param)

    @staticmethod
    def const_inflation(value):
        return lambda *args: value

    @staticmethod
    def conditional_inflation(value, thin_thresh):
        def _inflation(summary):
            if summary.thinness > thin_thresh:
                return summary.radius_det()
            else:
                return value

        return _inflation

###########Yuwei##########################
    @staticmethod
    def dynamic_inflation():
        def _inflation(summary):
            return summary.thinness

        return _inflation

###########Yuwei##########################

    @staticmethod
    def radius_inflation(delta=1e-4):
        return lambda summary: summary.radius_det(delta)

    @staticmethod
    def oful(d, alpha, radius):
        return Roful(d, alpha, SievedGreedyWorthFunction(radius, 1.0))

    @staticmethod
    def greedy(d, alpha):
        return Roful(d, alpha, GreedyWorthFunction())

    @staticmethod
    def sieved_greedy(d, alpha, radius, tolerance=1.0):
        return Roful(d, alpha, SievedGreedyWorthFunction(radius, tolerance))

    @property
    def d(self):
        return self.summary.d

    def choose_arm(self, ctx: Context) -> int:
        self.worth_func.bind(self.summary)  # mark
        values = self.worth_func.compute(ctx)
        return np.argmax(values).item()

    def update_model(self, feedback: Feedback):
        x = feedback.chosen_arm
        y = feedback.rew

        self.summary.add_obs(x, y)

    def update_metrics(self, feedback: Feedback):
        super().update_metrics(feedback)

        self.t += 1

        # outputs of interest
        self.thinnesses.append(self.summary.thinness)
        self.lambda_max.append(max(self.summary.scale))
        self.lambda_min.append(min(self.summary.scale))
        self.lambda_second.append(self.summary.scale[1])
        self.lambda_third.append(self.summary.scale[2])
        
        self.errors.append(np.linalg.norm(self.param - self.summary.mean)**2)

        weight_H = np.ones(self.d) / self.d
        weight_G = (self.param @ self.summary.basis)**2 / npl.norm(self.param)**2 
        print(weight_H)
        print(weight_G)
        
        gamma = self.d/self.t 
        
        
        def func(x):
            print(f"  gamma = {gamma}, ")
            func_ = lambda y:  self.func0(y,x, gamma)
            print(f"x = {x}")
            return 1-1/gamma -  self.discrete_integral(func_,  weight_H, self.summary.scale)

        c_0 = fsolve(func, 1e-3)

        func1_ = lambda x:  self.func1(x,c_0, gamma)
        func2_ = lambda x:  self.func2(x,c_0, gamma)
        
        int_H_1 = self.discrete_integral(func1_, c_0, weight_H, self.summary.scale)
        int_H_2 = self.discrete_integral(func2_, c_0, weight_H, self.summary.scale)
        int_G_1 = self.discrete_integral(func1_, c_0, weight_G, self.summary.scale)

        B_ = npl.norm(self.param)**2 * (1+ self.gamma * c_0 *int_H_2/ int_H_1 ) * int_G_1
        V_ = npl.norm(self.param)**2 * (1+ self.gamma * c_0 *int_H_2/ int_H_1 ) * int_H_1

        self.B_n.append(B_)
        self.V_n.append(V_)
        self.R_n.append(B_ + V_)



        

        self.outputs = ( self.metrics.regrets, self.thinnesses, self.errors, self.lambda_max, self.lambda_min, self.lambda_second, self.lambda_third,self.B_n, self.V_n)


    @staticmethod
    def discrete_integral( func, weight, support):
        # calulate the discrete integral of empirical distribution
        # weight: weight of each sample
        # support: support of the empirical distribution
        # func: function to be integrated
    
        # x:where to evaluate the integral
        # return: discrete integral of empirical distribution
        print(f"  support = {support} ")
        print(f" weight = {weight}")
        

        return np.sum(func(support) @ weight)

    def plot_hist(self, save_name):
        plt.clf()
        error = self.param - self.summary.mean
        plt.hist(error, bins=100)
        plt.savefig(save_name, dpi=600)


class ThinnessDirectedWorthFunction(ProductWorthFunction):
    compensator: np.ndarray

    def __init__(self, inflation, state=npr):
        self.inflation = inflation
        self.state = state

    @property
    def d(self):
        return self.summary.d

    def update(self):
        d = self.d
        rand = self.state.randn(d, 15)
        # print(rand.shape)

        basis = self.summary.basis
        scale = self.summary.scale
        # print(basis.shape)
        # print(scale.shape)

        # print((1/scale[np.newaxis,:]).shape)
        #print((1/scale ** 0.5 @rand  ).shape)
        self.compensator = (
            #self.inflation(self.summary) * basis.T @ (rand / scale ** 0.5)
            basis.T @ (np.multiply(1/scale[:, np.newaxis] ** 0.5, rand))
        )

        #self.all_rew = self.summary.all_rew
    def compute(self, ctx: Context) -> np.ndarray:

        values = ctx.arms @ self.candidates()
        # print(values.shape)
        # print("lol")
        regret = values.max(axis=0, keepdims=True) - values
        regret = np.mean(regret, axis=1)
        # print(regret.shape)
        svd_list = [npl.svd(self.summary.xx + np.outer(arm, arm),
                            hermitian=True)for arm in ctx.arms]
        thinness_list = np.array(
            [(max(1/svd_[1]) * self.d / sum(1/svd_[1])) ** 0.5 for svd_ in svd_list])
        thinness_list_delta = self.summary.thinness - thinness_list
        # print(self.summary.thinness)
        #svdd = npl.svd( self.summary.xx)
        #print((max(1/svdd[1]) * self.d / sum(1/svdd[1])) ** 0.5)
        # print(thinness_list_delta)
        #values = -(regret**2) * thinness_list
        values = -(regret**2) * (1 + np.exp(- thinness_list_delta))
        # print(values)
        #values = np.minimum(values, np.zeros_like(values))
        # print(values)
        #values = np.array(values)
        # print(values.shape)
        # print(values.ndim)

        if values.ndim == 2:
            values = np.max(values, axis=1)

        return values

    def candidates(self):
        return self.summary.mean[:, np.newaxis] + self.compensator


class PureThinnessWorthFunction(ProductWorthFunction):
    compensator: np.ndarray

    def __init__(self, inflation, state=npr):
        self.inflation = inflation
        self.state = state

    @property
    def d(self):
        return self.summary.d

    def update(self):
        d = self.d
        rand = self.state.randn(d)

        basis = self.summary.basis
        scale = self.summary.scale

        self.compensator = (
            #self.inflation(self.summary) * basis.T @ (rand / scale ** 0.5)
            basis.T @ (rand / scale ** 0.5)
        )

        #self.all_rew = self.summary.all_rew
    def compute(self, ctx: Context) -> np.ndarray:

        #values = ctx.arms @ self.candidates()
        svd_list = [npl.svd(self.summary.xx + np.outer(arm, arm),
                            hermitian=True)for arm in ctx.arms]
        thinness_list = [(max(1/svd_[1]) * self.d / sum(1/svd_[1]))
                         ** 0.5 for svd_ in svd_list]
        values = [1/thinness_ for thinness_ in thinness_list]
        values = np.array(values)
        # print(values.shape)

        if values.ndim == 2:
            values = np.max(values, axis=1)

        return values

    def candidates(self):
        return self.summary.mean + self.compensator


class TsWorthFunction(ProductWorthFunction):
    compensator: np.ndarray

    def __init__(self, inflation, state=npr):
        self.inflation = inflation
        self.state = state

    @property
    def d(self):
        return self.summary.d

    def update(self):
        d = self.d
        rand = self.state.randn(d)

        basis = self.summary.basis
        scale = self.summary.scale

        self.compensator = (
            self.inflation(self.summary) * basis.T @ (rand / scale ** 0.5)
        )

    def candidates(self):
        return self.summary.mean + self.compensator


class GreedyWorthFunction(ProductWorthFunction):
    def __init__(self, inflation=1.0):
        self.inflation = inflation

    def candidates(self):
        return self.summary.mean


class SievedGreedyWorthFunction(WorthFunction):
    radius: Callable
    tolerance: float

    def __init__(self, radius, tolerance):
        self.radius = radius
        self.tolerance = tolerance

    def compute(self, ctx: Context) -> np.ndarray:
        lowers, centers, uppers = self.confidence_bounds(ctx.arms)

        # sieving arms
        baseline = lowers.max()
        optimal = uppers.max()

        threshold = self.tolerance * optimal + \
            (1.0 - self.tolerance) * baseline
        survivors = uppers >= threshold

        # computing the values
        return np.where(survivors, centers, lowers)

    def confidence_center(self, arms):
        return arms @ self.summary.mean

    def confidence_width(self, arms):
        scale = arms @ npl.solve(self.summary.xx, arms.T)

        if len(scale.shape) == 2:
            scale = np.diag(scale)

        return self.radius() * scale ** 0.5

    def confidence_bounds(self, arms):
        centers = self.confidence_center(arms)
        widths = self.confidence_width(arms)

        return centers - widths, centers, centers + widths
