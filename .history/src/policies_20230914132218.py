import abc
from locale import normalize

from typing import List, Callable

import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt
import math

from envs import Context
from envs import Feedback
from utils import DataSummary
from utils import predicted_risk
from utils import timing
from utils import timing_block

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import fsolve

import cvxpy as cp
#import gurobipy


class Metric:
    regrets: List[float]

    def __init__(self):
        self.regrets = []
        self.chosen_arms = []

    def add_obs(self, feedback: Feedback):
        self.regrets.append(feedback.regret)
        self.chosen_arms.append(feedback.regret)


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

    summary: DataSummary
    worth_func: WorthFunction

    def __init__(self, d, lambda_, worth_func: WorthFunction,  param=None, noise_sd = None, t = None, radius = None, delta = None, inflation = None):
        super().__init__()

        self.t = t
        self.radius = radius
        self.delta = delta
        self.inflation = inflation
        
        self.tt = 0


        self.summary = DataSummary(d, lambda_, noise_sd, radius, param, inflation,self.t)
        self.worth_func = worth_func

        self.param = param
        self.noise_sd = noise_sd

        self.func0 = lambda x, c_0, gamma:  1/(1+ c_0 * gamma * x)
        self.func1 = lambda x, c_0, gamma:  x/(1+ c_0 * gamma * x)**2
        self.func2 = lambda x, c_0, gamma:  x**2/(1+ c_0 * gamma * x)**2

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
        






    @staticmethod
    def ts(d, lambda_, inflation=1.0, state=None, param=None, noise_sd = None, t = None, radius = None, delta = None):
        if isinstance(inflation, float):
            inflation = Roful.const_inflation(inflation)
        
            

        return Roful(d, lambda_, TsWorthFunction(inflation, delta = delta, state=state), param, noise_sd = noise_sd, t = t, radius = radius, delta = delta, inflation = inflation)

    @staticmethod
    def oful(d, lambda_, radius_oful, param=None, noise_sd = None, t = None, radius = None, delta = None):
        return Roful(d, lambda_, SievedGreedyWorthFunction(radius_oful, 1.0), param=param, noise_sd = noise_sd, t = t, radius = radius, delta = delta, inflation = Roful.const_inflation(1.0))

    @staticmethod
    def spects(d, lambda_, inflation=1.0, state=None, param=None, noise_sd = None, t = None, radius = None, delta = None, alpha = None, radius_oful = None):
        if isinstance(inflation, float):
            inflation = Roful.const_inflation(inflation)

        return Roful(d, lambda_, SgTsWorthFunction(inflation, alpha = alpha, radius = radius, noise_sd = noise_sd, delta = delta,radius_oful = radius_oful, tolerance = 1, state=state), param, noise_sd = noise_sd, t = t,  radius = radius, delta = delta, inflation = inflation)


    @staticmethod
    def greedy(d, lambda_, param=None, noise_sd = None, t = None, radius = None, delta = None):
        return Roful(d, lambda_, GreedyWorthFunction(), param = param, noise_sd = noise_sd, t = t, radius = radius, delta = delta, inflation = Roful.const_inflation(1))

    @staticmethod
    def sieved_greedy(d, alpha, radius, tolerance=1.0):
        return Roful(d, alpha, SievedGreedyWorthFunction(radius, tolerance))

    def pure_thin(d, lambda_, inflation=1.0, state=None):
        if isinstance(inflation, float):
            inflation = Roful.const_inflation(inflation)

        return Roful(d, lambda_, PureThinnessWorthFunction(inflation, state=state))

    def thin_dirt(d, lambda_, inflation=1.0, state=None, param=None, noise_sd = None, t = None, radius = None):
        if isinstance(inflation, float):
            inflation = Roful.const_inflation(inflation)

        return Roful(d, lambda_, ThinnessDirectedWorthFunction(inflation, state=state), param, noise_sd = noise_sd, t = t, radius = radius)

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


    @staticmethod
    def radius_OFUL(delta=1e-3):
        return lambda summary: summary.radius_OFUL(delta)


    @staticmethod
    def radius_inflation(delta=1e-3):
        return lambda summary: summary.radius_OFUL(delta)

    @staticmethod
    def dynamic_inflation():
        def _inflation(summary):
            return summary.thinness

    @staticmethod
    def ts_freq_inflation(delta=1e-3):
        return lambda summary: summary.radius_normal(delta)

    @staticmethod
    def ts_inflation_alter(delta=1e-3):

        return lambda summary: summary.noise_sd * summary.radius_normal(delta) / summary.radius_det(delta) 


    @property
    def d(self):
        return self.summary.d

    def choose_arm(self, ctx: Context) -> int:
        self.worth_func.bind(self.summary)  # mark

        values = self.worth_func.compute(ctx)
        return np.argmax(values).item()


    def exploration_amount(self, arms):
        scale = arms @ npl.solve(self.summary.xx, arms.T)

        if len(scale.shape) == 2:
            scale = np.diag(scale)

        return scale ** 0.5

    def calculate_oracle_alpha(self, feedback : Feedback):
        ctx = feedback.ctx
        expl = self.exploration_amount(ctx.arms)
        
        
        true_value = ctx.arms @ self.param
        if expl[np.argmax(true_value).item()]  ==0 and expl[feedback.arm_idx] == 0:
            oracle_alpha = 1.0
        else:
            oracle_alpha =  expl[feedback.arm_idx]/ expl[np.argmax(true_value).item()]

        
        return oracle_alpha

    #@timing
    def update_model(self, feedback: Feedback):
        x = feedback.chosen_arm
        y = feedback.rew

        self.summary.add_obs(x, y)

    #@timing
    def update_metrics(self, feedback: Feedback):
        super().update_metrics(feedback)

        self.tt = self.tt + 1
        self.bad = 0 


        # with timing_block('appending'):
        lambda_max = max(self.summary.scale)
        lambda_min = min(self.summary.scale)


        beta = (self.radius* np.sqrt(self.summary.lambda_) \
            + self.noise_sd * np.sqrt(2 * (self.d/2)*np.log((self.summary.lambda_ + self.t))\
            +2*  ( - self.d/2)*np.log( self.summary.lambda_ ) - 2*np.log(self.delta/2/self.t )))

        
        self.beta = self.summary.radius_det()
        


        weight_G = ( np.array(self.summary.basis)@np.array(self.param))**2 / npl.norm(self.param)**2 

        theta_hat = self.summary.mean 


        weight_theta_hat =  ( np.array(self.summary.basis)@np.array(theta_hat)) 
        normalized_weight_theta_hat = weight_theta_hat / npl.norm(theta_hat) 
        
        weight_diff = (beta/ self.summary.scale ** 0.5  * np.sign(weight_theta_hat) )

        weight_theta_hat_upper =  weight_theta_hat +  weight_diff


        
        zeta1 = theta_hat.T @ self.summary.xx @ theta_hat
        zeta2 = beta**2
        zeta_square = theta_hat.T @ self.summary.xx @ theta_hat - beta**2
        zeta = np.sqrt(np.maximum(zeta_square, 0 ))


        if self.tt == self.t:

            print(f"bad ratio = {self.bad/self.tt}")


        '''
        scale_ = self.summary.scale/ self.tt

        def func(x):
            #print(f"  gamma = {gamma}, ")
            func_ = lambda y:  self.func0(y,x, gamma)
            #print(f"c0 = {x}")
            return 1-1/gamma -  self.discrete_integral(func_,  weight_H, scale_)

        c_0 = fsolve(func, 1e-3)

        func_ = lambda y:  self.func0(y,c_0, gamma)
        #print(f"t = {self.t}, tt = {self.tt}, gamma = {gamma}, c_0 = {c_0}, obj_value = {1-1/gamma -  self.discrete_integral(func_,  weight_H, scale_)}")

        func1_ = lambda x:  self.func1(x,c_0, gamma)
        func2_ = lambda x:  self.func2(x,c_0, gamma)
        
        int_H_1 = self.discrete_integral(func1_, weight_H, scale_)
        int_H_2 = self.discrete_integral(func2_, weight_H, scale_)
        int_G_1 = self.discrete_integral(func1_, weight_G, scale_)

        B_ = npl.norm(self.param)**2 * (1+ gamma * c_0 *int_H_2/ int_H_1 ) * int_G_1
        V_ = self.noise_sd**2  * gamma * c_0 *int_H_2/ int_H_1 

        self.B_n.append(B_[0])
        self.V_n.append(V_[0])
        self.R_n.append((B_[0] + V_[0]))
        '''



        ## append the results to the list
        self.thinnesses.append(self.summary.thinness)
        self.lambda_max.append(lambda_max)
        self.lambda_min.append(lambda_min)

        self.lambda_max_over_min.append(lambda_max/lambda_min)
        self.errors.append(np.linalg.norm(self.param - self.summary.mean)**2)

        self.worst_alpha.append(np.sqrt(lambda_min/lambda_max))
        
        self.zeta.append(zeta)
        self.proj_first.append(theta_hat.T @ self.summary.xx @ theta_hat/lambda_max/npl.norm(theta_hat)**2)


        self.approx_alpha.append(self.worth_func.alpha_approx)

        beta_TS = self.inflation(self.summary) * self.summary.radius_det()

        assert((self.beta+beta_TS)/self.beta == (self.summary.radius_TS()+self.summary.radius_det())/self.summary.radius_det())

        
        self.iota.append(self.inflation(self.summary))
        self.betas_TS.append(beta_TS)
        self.betas.append(self.beta)
        

        oracle_alpha = self.calculate_oracle_alpha(feedback)


        self.outputs = ( self.worth_func.alphas, self.worth_func.mus, self.worst_alpha, self.metrics.regrets, self.thinnesses, self.iota, self.lambda_max, self.lambda_min,  self.proj_first,  self.betas, self.betas_TS, self.worth_func.discrete_alphas)







    @staticmethod
    def solve_LP (XX_scale, XX_norm_lower, XX_norm_upper):
        n = XX_scale.shape[0]
        x = cp.Variable(n)
        XX_scale_inv = 1/XX_scale

        constraints = [x>=0, cp.norm(x,1) <= 1,  x@XX_scale <=  XX_norm_upper,  x@XX_scale  >= XX_norm_lower]

        prob = cp.Problem(cp.Maximize( x@XX_scale_inv ), constraints)
        prob.solve(solver=cp.GUROBI)
        

        return np.sqrt(prob.value), x.value

    @staticmethod
    def solve_LP2 (XX_scale, center, beta):
        n = XX_scale.shape[0]
        x = cp.Variable(n)
        XX_scale_inv = 1/XX_scale

        constraints = [  x**2 @ XX_scale <=  beta**2 ]

        prob = cp.Problem(cp.Minimize( cp.norm(x + center,2) ), constraints)
        prob.solve(solver=cp.GUROBI)
        

        return prob.value, x.value




class GradientDirectedWorthFunction(ProductWorthFunction):
    compensator: np.ndarray

    def __init__(self, inflation, state=npr):
        self.inflation = inflation
        self.state = state
        self.alphas = []

    @property
    def d(self):
        return self.summary.d

    def update(self):
        d = self.d
        rand = self.state.randn(d, 15)

        basis = self.summary.basis
        scale = self.summary.scale

        self.xx_inv = basis.T * 1 / scale@  basis


        # print((1/scale[np.newaxis,:]).shape)
        #print((1/scale ** 0.5 @rand  ).shape)
        self.compensator = (
            #self.inflation(self.summary) * basis.T @ (rand / scale ** 0.5)
            basis.T @ (np.multiply(1/scale[:, np.newaxis] ** 0.5, rand)) 
        )

    def compute(self, ctx: Context) -> np.ndarray:

        values = ctx.arms @ self.candidates()




        regret = values.max(axis=0, keepdims=True) - values
        regret = np.mean(regret, axis=1)

        grad_norm_list = [ np.sqrt(arm.T @ self.xx_inv @ arm/ npl.norm(theta_hat)**2)  for arm in ctx.arms]


        svd_list = [npl.svd(self.summary.xx + np.outer(arm, arm),
                            hermitian=True)for arm in ctx.arms]
        thinness_list = np.array(
            [(max(1/svd_[1]) * self.d / sum(1/svd_[1])) ** 0.5 for svd_ in svd_list])
        thinness_list_delta = self.summary.thinness - thinness_list

        values = -(regret**2) * (1 + np.exp(- thinness_list_delta))

        if values.ndim == 2:
            values = np.max(values, axis=1)

        return values

    def candidates(self):
        return self.summary.mean[:, np.newaxis] + self.compensator #


class ThinnessDirectedWorthFunction(ProductWorthFunction):
    compensator: np.ndarray

    def __init__(self, inflation, state=npr):
        self.inflation = inflation
        self.state = state
        self.alphas = []

    @property
    def d(self):
        return self.summary.d

    def update(self):
        d = self.d
        rand = self.state.randn(d, 15)

        basis = self.summary.basis
        scale = self.summary.scale

        self.compensator = (
            basis.T @ (np.multiply(1/scale[:, np.newaxis] ** 0.5, rand))
        )

    def compute(self, ctx: Context) -> np.ndarray:

        values = ctx.arms @ self.candidates()

        regret = values.max(axis=0, keepdims=True) - values
        regret = np.mean(regret, axis=1)

        svd_list = [npl.svd(self.summary.xx + np.outer(arm, arm),
                            hermitian=True)for arm in ctx.arms]
        thinness_list = np.array(
            [(max(1/svd_[1]) * self.d / sum(1/svd_[1])) ** 0.5 for svd_ in svd_list])
        thinness_list_delta = self.summary.thinness - thinness_list

        values = -(regret**2) * (1 + np.exp(- thinness_list_delta))


        if values.ndim == 2:
            values = np.max(values, axis=1)

        return values

    def candidates(self):
        return self.summary.mean[:, np.newaxis] + self.compensator #


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

    def compute(self, ctx: Context) -> np.ndarray:

        svd_list = [npl.svd(self.summary.xx + np.outer(arm, arm),
                            hermitian=True)for arm in ctx.arms]
        thinness_list = [(max(1/svd_[1]) * self.d / sum(1/svd_[1]))
                         ** 0.5 for svd_ in svd_list]
        values = [1/thinness_ for thinness_ in thinness_list]
        values = np.array(values)

        if values.ndim == 2:
            values = np.max(values, axis=1)

        return values

    def candidates(self):
        return self.summary.mean + self.compensator


class TsWorthFunction(ProductWorthFunction):
    compensator: np.ndarray
    

    def __init__(self, inflation, delta,state=npr):
        self.inflation = inflation
        self.state = state
        self.alphas = []
        self.mus = []
        self.delta = delta
        self.discrete_alphas = []


    @property
    def d(self):
        return self.summary.d

    def update(self):
        d = self.d
        rand = self.state.randn(d)

        basis = self.summary.basis
        scale = self.summary.scale

        inv_principle_scale  = np.zeros_like(scale)
        inv_principle_scale[0] =1/self.summary.scale[0] 
        
        self.radius_det = self.summary.radius_det()
        self.radius_TS = self.summary.radius_TS()


        self.compensator = (
            self.radius_TS  * basis.T @ (rand / scale ** 0.5)  /  self.summary.radius_normal(self.delta)
        )


        
    def candidates(self):
        return self.summary.mean + self.compensator

    def principle_candidates(self):
        return self.summary.mean + self.principle_compensator

    def compute(self, ctx: Context) -> np.ndarray:
        discrete_alpha = self.summary.calculate_discrete_alpha(ctx.arms)
        self.discrete_alpha = discrete_alpha

        values = ctx.arms @ self.candidates()
        self.discrete_alphas.append(self.discrete_alpha)

        

        if values.ndim == 2:
            values = np.max(values, axis=1)
            
        self.alpha_approx = self.summary.calculate_alpha
        
        
        discrete_mu = (self.summary.inflation(self.summary)+1.0) * (1 + discrete_alpha)
        #self.mu_approx = self.summary.calculate_mu
        self.mu_approx = discrete_mu
        self.alphas.append(self.alpha_approx)
        self.mus.append(self.mu_approx)
        

        return values



class GreedyWorthFunction(ProductWorthFunction):
    def __init__(self, inflation=1.0):
        self.inflation = inflation
        self.alphas = []
        self.mus = []

    def candidates(self):
        return self.summary.mean

    def compute(self, ctx: Context) -> np.ndarray:
        values = ctx.arms @ self.candidates()


        if values.ndim == 2:
            values = np.max(values, axis=1)
        self.alpha_approx = self.summary.calculate_alpha
        self.mu_approx = self.summary.calculate_mu
        self.alphas.append(self.alpha_approx)
        self.mus.append(self.mu_approx)

        return values
    
class SievedGreedyWorthFunction(WorthFunction):
    radius: Callable
    tolerance: float

    def __init__(self, radius, tolerance):
        self.radius = radius
        self.tolerance = tolerance
        self.alphas = []
        self.mus = []
        self.discrete_alphas = []
        

    def compute(self, ctx: Context) -> np.ndarray:
        lowers, centers, uppers = self.confidence_bounds(ctx.arms)

        # sieving arms
        baseline = lowers.max()
        optimal = uppers.max()
        minus_infty = np.ones_like(centers) * (-np.inf)

        threshold = self.tolerance * optimal + \
            (1.0 - self.tolerance) * baseline
        survivors = uppers >= threshold

        # computing the values
        assert( self.tolerance == 1.0)
        self.radius_TS = self.summary.radius_TS()
        self.radius_det = self.summary.radius_det()
        discrete_alpha = self.summary.calculate_discrete_alpha(ctx.arms)

        discrete_mu = (self.summary.inflation(self.summary)+1.0) * (1 + discrete_alpha)
        
        if self.tolerance == 1.0:
            self.mu_approx = 2.0
            self.discrete_alpha = 1.0
            self.discrete_alphas.append(self.discrete_alpha)
            self.alpha_approx = 1.0
            #self.alpha_approx = (self.radius_TS+self.radius_det)/self.radius_det
            
            #print(f"radius_TS_in = {self.radius_TS}, radius_det_in = {self.summary.radius_det()}, alpha_approx = {self.alpha_approx}")
            self.alphas.append(  self.alpha_approx  )
            self.mus.append(self.mu_approx)
        else:
            self.discrete_alpha = discrete_alpha
            self.discrete_alphas.append(self.discrete_alpha)
            self.alpha_approx = self.summary.calculate_alpha
            #self.mu_approx = self.summary.calculate_mu
            self.mu_approx =discrete_mu
            self.alphas.append(self.alpha_approx)
            self.mus.append(self.mu_approx)

        
        

        return np.where(survivors, centers, minus_infty)

    def confidence_center(self, arms):
        #print(f"confidence_center = {arms @ self.summary.mean}")

        return arms @ self.summary.mean
        
    def confidence_width(self, arms):
        scale = arms @ npl.solve(self.summary.xx, arms.T)
        

        if len(scale.shape) == 2:
            scale = np.diag(scale)

        return self.radius(self.summary) * scale ** 0.5

    def confidence_bounds(self, arms):
        centers = self.confidence_center(arms)
        widths = self.confidence_width(arms)
        return centers - widths, centers, centers + widths

class SgTsWorthFunction(WorthFunction):
    compensator: np.ndarray

    def __init__(self, inflation, alpha, radius, noise_sd, delta,radius_oful, tolerance, state=npr):
        self.inflation = inflation
        self.state = state
        self.alpha = alpha
        self.radius_oful = radius_oful
        self.tolerance = tolerance
        self.radius = radius
        self.noise_sd = noise_sd
        self.delta = delta
        self.alphas = []
        self.mus = []
        self.discrete_alphas = []


    @property
    def d(self):
        return self.summary.d
    @property
    def t(self):
        return self.summary.t


    def compute(self, ctx: Context) -> np.ndarray:
        values = ctx.arms @ self.candidates()
        #true_value = ctx.arms @ self.summary.param
        mean_value = ctx.arms @ self.summary.mean
        lowers, centers, uppers = self.confidence_bounds(ctx.arms)
        exp_amount = self.exploration_amount(ctx.arms)


        if values.ndim == 2:
            values = np.max(values, axis=1)
            
        minus_infty = np.ones_like(values) * (-np.inf)
        threshold = self.alpha * np.max(exp_amount)
        survivors = exp_amount >= threshold
        #alpha_approx = self.calculate_alpha  
        alpha_approx = self.summary.calculate_alpha
        mu_approx = self.summary.calculate_mu
        
        discrete_alpha = self.summary.calculate_discrete_alpha(ctx.arms)
        discrete_mu = (self.summary.inflation(self.summary)+1.0) * (1 + discrete_alpha)
        

        if self.t == math.ceil(self.d/2*3) + 1:
            

            print(f"t = {self.t},d = {self.d}")

        if discrete_mu < self.alpha:
            self.alpha_approx = alpha_approx
            self.alphas.append(self.alpha_approx)
            self.mu_approx = discrete_mu
            self.mus.append (self.mu_approx)

            self.discrete_alpha = discrete_alpha
            self.discrete_alphas.append(self.discrete_alpha)
            return values
        else:
            self.mu_approx = 2.0
            self.discrete_alpha = discrete_alpha
            self.alpha_approx = alpha_approx


            #return np.where(survivors, values, minus_infty)
            self.alphas.append(self.alpha_approx)
            
            self.mus.append (self.mu_approx)

            self.discrete_alphas.append(self.discrete_alpha)


            return uppers



    def update(self):
        d = self.d
        rand = self.state.randn(d)

        basis = self.summary.basis
        scale = self.summary.scale

        inv_principle_scale  = np.zeros_like(scale)
        inv_principle_scale[0] =1/self.summary.scale[0] 
        
        self.radius_det = self.summary.radius_det()
        self.radius_TS = self.summary.radius_TS()
        self.compensator = (
            self.radius_TS * basis.T @ (rand / scale ** 0.5)  /  self.summary.radius_normal(self.delta)

        )

        self.beta =  self.summary.radius_det()


    def candidates(self):

        return self.summary.mean + self.compensator

    def confidence_center(self, arms):
        return arms @ self.summary.mean

    def confidence_width(self, arms):
        scale = arms @ npl.solve(self.summary.xx, arms.T)

        if len(scale.shape) == 2:
            scale = np.diag(scale)

        return self.radius_oful(self.summary) * scale ** 0.5

    def confidence_bounds(self, arms):
        centers = self.confidence_center(arms)
        widths = self.confidence_width(arms)

        return centers - widths, centers, centers + widths

    def exploration_amount(self, arms):
        scale = arms @ npl.solve(self.summary.xx, arms.T)

        if len(scale.shape) == 2:
            scale = np.diag(scale)

        return scale ** 0.5




