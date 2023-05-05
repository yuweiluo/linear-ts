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
    # Yuwei

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
        self.all_rew = 0

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
    # added param as input to calculate error
    def ts(d, lambda_, inflation=1.0, state=None, param=None, noise_sd = None, t = None, radius = None, delta = None):
        if isinstance(inflation, float):
            inflation = Roful.const_inflation(inflation)
        
            

        return Roful(d, lambda_, TsWorthFunction(inflation, state=state), param, noise_sd = noise_sd, t = t, radius = radius, delta = delta, inflation = inflation)

    @staticmethod
    def oful(d, lambda_, radius_SG, param=None, noise_sd = None, t = None, radius = None, delta = None):
        return Roful(d, lambda_, SievedGreedyWorthFunction(radius_SG, 1.0), param=param, noise_sd = noise_sd, t = t, radius = radius, delta = delta, inflation = Roful.const_inflation(1.0))

    @staticmethod
    # added param as input to calculate error
    def spects(d, lambda_, inflation=1.0, state=None, param=None, noise_sd = None, t = None, radius = None, delta = None, alpha = None, radius_oful = None):
        if isinstance(inflation, float):
            inflation = Roful.const_inflation(inflation)

        return Roful(d, lambda_, SgTsWorthFunction(inflation, alpha = alpha, radius = radius, noise_sd = noise_sd, delta = delta,radius_oful = radius_oful, tolerance = 1, state=state), param, noise_sd = noise_sd, t = t,  radius = radius, delta = delta, inflation = inflation)





    @staticmethod
    def greedy(d, lambda_, param=None, noise_sd = None, t = None, radius = None, delta = None):
        return Roful(d, lambda_, GreedyWorthFunction(), param = param, noise_sd = noise_sd, t = t, radius = radius, delta = delta, inflation = Roful.const_inflation(1))

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

###########Yuwei##########################

    @staticmethod
    def dynamic_inflation():
        def _inflation(summary):
            return summary.thinness



###########Yuwei##########################

    @staticmethod
    def radius_inflation(delta=1e-3):
        return lambda summary: summary.radius_det(delta)






    @staticmethod
    def sieved_greedy(d, alpha, radius, tolerance=1.0):
        return Roful(d, alpha, SievedGreedyWorthFunction(radius, tolerance))

    @property
    def d(self):
        return self.summary.d

    def choose_arm(self, ctx: Context) -> int:
        self.worth_func.bind(self.summary)  # mark

        #print(f"candidate = {self.summary.mean + self.worth_func.compensator}, param = {self.param}")
        #print(f"xx = {self.summary.xx }")

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
        #print(f"expl = {expl}, chosen_arm = {feedback.chosen_arm}, valid =  {ctx.arms[feedback.arm_idx]},ture_index = {np.argmax(true_value).item()}")
        if expl[np.argmax(true_value).item()]  ==0 and expl[feedback.arm_idx] == 0:
            oracle_alpha = 1.0
        else:
            oracle_alpha =  expl[feedback.arm_idx]/ expl[np.argmax(true_value).item()]

        
        return oracle_alpha
        
        


    @property
    def calculate_alpha(self):
        
        theta_hat =  self.summary.mean
        theta_hat_Vt_norm_sqr = theta_hat.T @ self.summary.xx @ theta_hat
        lambda_max = max(self.summary.scale)
        lambda_min = min(self.summary.scale)
        

        if theta_hat_Vt_norm_sqr >= self.beta**2:
            
            zeta = (theta_hat_Vt_norm_sqr - self.beta**2)**0.5
            #theta_norm_lower_bound = npl.norm(theta_hat) - beta/np.sqrt()
            theta_norm_upper_bound = np.maximum( npl.norm(theta_hat - self.beta/np.sqrt(lambda_min) * self.summary.basis[self.summary.d - 1]),npl.norm(theta_hat + self.beta/np.sqrt(lambda_min) * self.summary.basis[self.summary.d - 1]))
            theta_norm_upper_bound = npl.norm(theta_hat) + self.beta/np.sqrt(lambda_min)

            
            x_XXnorm_max  =  np.sqrt(theta_hat.T @ self.summary.xx @ self.summary.xx @ theta_hat)/zeta
            x_XXnorm_min  =  zeta / theta_norm_upper_bound

            opt_x_XXinvnorm_sqr, wst_x_XXinvnorm_sqr = self.new_bound(self.summary.scale, np.maximum(x_XXnorm_min**2,lambda_min), np.minimum( x_XXnorm_max**2,lambda_max))
            alpha = np.sqrt(wst_x_XXinvnorm_sqr/opt_x_XXinvnorm_sqr) 

        else:
            #print("badd!")
            #opt_x_XXinvnorm_sqr, wst_x_XXinvnorm_sqr = (1/lambda_min,1/lambda_max) 
            alpha = 0
        return alpha


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
        
        # outputs of interest

        # with timing_block('appending'):
        lambda_max = max(self.summary.scale)
        lambda_min = min(self.summary.scale)


        beta = (self.radius* np.sqrt(self.summary.lambda_) \
            + self.noise_sd * np.sqrt(2 * (self.d/2)*np.log((self.summary.lambda_ + self.t))\
            +2*  ( - self.d/2)*np.log( self.summary.lambda_ ) - 2*np.log(self.delta/2/self.t )))
        #beta = self.radius* np.sqrt(self.summary.lambda_) + self.noise_sd * np.sqrt(2 * np.log((self.summary.lambda_ + self.t)**(self.d/2)* self.summary.lambda_ ** ( self.d/2)/ self.delta  ))
        
        #self.beta = beta
        self.beta = self.summary.radius_det()
        #print(f"tt= {self.tt}, beta = {beta}, radius = {self.beta}")
        
        '''
        def radius_det(self, delta=1e-3):
            term1 = np.log(self.scale / self.lambda_).sum() - 2 * np.log(delta)
            term2 = self.lambda_ * self.param_bound ** 2

            return self.prior_var**0.5 * term1 ** 0.5 + term2 ** 0.5
        
        '''

        weight_G = ( np.array(self.summary.basis)@np.array(self.param))**2 / npl.norm(self.param)**2 

        theta_hat = self.summary.mean 
        #print(f"{theta_hat}{beta}{((self.summary.lambda_ + self.t)**(self.d/2)* self.summary.lambda_ ** ( - self.d/2)  )}")

        weight_theta_hat =  ( np.array(self.summary.basis)@np.array(theta_hat)) 
        normalized_weight_theta_hat = weight_theta_hat / npl.norm(theta_hat) 
        
        weight_diff = (beta/ self.summary.scale ** 0.5  * np.sign(weight_theta_hat) )
        #opt_norm , weight_diff = self.solve_LP2(self.summary.scale,weight_theta_hat, beta)
            #self.approx_alpha.append(self.x_XXinvnorm_min/opt_XX_inv_norm) 
        weight_theta_hat_upper =  weight_theta_hat +  weight_diff

        #print(f"weight_theta_hat_upper = {npl.norm(weight_theta_hat_upper)}, opt_norm = {opt_norm}")
        
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
        #self.lambda_second.append(self.summary.scale[1])
        #self.lambda_third.append(self.summary.scale[2])
        #self.lambda_third.append(0)
        #self.lambda_d_minus_1.append(self.summary.scale[self.d-2])
        #self.lambda_half_d.append(self.summary.scale[self.d//2])
        self.lambda_max_over_min.append(lambda_max/lambda_min)
        self.errors.append(np.linalg.norm(self.param - self.summary.mean)**2)
        #self.oracle_regrets.append(feedback.oracle_regret) 
        #self.errors_candidate.append(np.linalg.norm(self.param - self.worth_func.candidates())**2)
        #self.errors_pcandidate.append(np.linalg.norm(self.param - self.worth_func.principle_candidates())**2)
        self.worst_alpha.append(np.sqrt(lambda_min/lambda_max))
        
        self.zeta.append(zeta)
        self.proj_first.append(theta_hat.T @ self.summary.xx @ theta_hat/lambda_max/npl.norm(theta_hat)**2)
        #self.proj_first.append(self.param.T @ self.summary.xx @ self.param/lambda_max/npl.norm(self.param)**2)
        
        #self.approx_alpha.append(self.calculate_alpha)
        #self.approx_alpha.append(self.summary.calculate_alpha)
        self.approx_alpha.append(self.worth_func.alpha_approx)
        beta_TS = self.inflation(self.summary) * np.sqrt(2*self.d*np.log(2*self.d/self.delta*2*self.t))
        #print(f"inflation_TS = {self.inflation(self.summary) }" )
        print(f"beta_TS = {beta_TS}, radius_TS = {self.summary.radius_TS()},")
        print(f"beta = {self.beta}, radius_det = {self.summary.radius_det()}")
        #print(f"ratio1 = {(self.beta+beta_TS)/self.beta}, ratio2 = { (self.summary.radius_TS()+self.summary.radius_det())/self.summary.radius_det()}")
        #print(f"T = {self.t}, delta = {self.delta}, ")
        #print(f"oful_ratio_Inv = {beta_TS/(self.beta+beta_TS)}")
        #Aassert((self.beta+beta_TS)/self.beta == (self.summary.radius_TS()+self.summary.radius_det())/self.summary.radius_det())
        #assert(self.worth_func.alphas[-1] == self.worth_func.alpha_approx)
        self.oracle_alpha.append(self.beta/(self.beta+beta_TS)*self.worth_func.alpha_approx)
        #print(f"self.oracle_alpha = {self.oracle_alpha[-1]}, {self.beta/(self.summary.radius_TS()+self.summary.radius_det())*self.worth_func.alpha_approx}")
        #print(f"self.worth_func.alpha_approx = {self.worth_func.alpha_approx}, other = {(self.beta+beta_TS)/self.beta}")
        self.iota.append(self.inflation(self.summary))
        self.betas_TS.append(beta_TS)
        self.betas.append(self.beta)
        
        #print(f"self.calculate_alpha = {self.calculate_alpha}, new_alpha = {self.summary.calculate_alpha} ")
        oracle_alpha = self.calculate_oracle_alpha(feedback)
        #print(f"oracle_alpha = {oracle_alpha}")


        self.outputs = ( self.worth_func.alphas, self.oracle_alpha, self.worst_alpha, self.metrics.regrets, self.thinnesses, self.errors, self.lambda_max, self.lambda_min,  self.proj_first,  self.betas, self.betas_TS)





    @staticmethod
    def new_bound (scale, XX_norm_sqr_min, XX_norm_sqr_max):
        lambda_max = np.max(scale)
        lambda_min = np.min(scale)

        lambda_up = np.min(scale[scale>=XX_norm_sqr_max])
        lambda_down = np.max(scale[scale<=XX_norm_sqr_max])
        #print(f"lambda_up = {lambda_up}, lambda_down = {lambda_down}, lambda_max = {lambda_max}, lambda_min = {lambda_min}")
        # opt, wst
        return (1/lambda_max + 1/lambda_min - XX_norm_sqr_min/lambda_max/lambda_min), (1/lambda_up + 1/lambda_down - XX_norm_sqr_max/lambda_up/lambda_down)

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

    @staticmethod
    def discrete_integral( func, weight, support):
        # calulate the discrete integral of empirical distribution
        # weight: weight of each sample
        # support: support of the empirical distribution
        # func: function to be integrated
    
        # x:where to evaluate the integral
        # return: discrete integral of empirical distribution

        return np.sum(func(support) @ weight)

    def plot_hist(self, save_name):
        plt.clf()
        error = self.param - self.summary.mean
        plt.hist(error, bins=100)
        plt.savefig(save_name, dpi=600)


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
        # print(rand.shape)

        basis = self.summary.basis
        scale = self.summary.scale

        self.xx_inv = basis.T * 1 / scale@  basis


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

        regret = values.max(axis=0, keepdims=True) - values
        regret = np.mean(regret, axis=1)

        grad_norm_list = [ np.sqrt(arm.T @ self.xx_inv @ arm/ npl.norm(theta_hat)**2)  for arm in ctx.arms]

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
        self.alphas = []


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
        

        #print(f"scale = {scale}") 
        #print(f"self.summary.scale[0] = {self.summary.scale[0]}")
        #print(f"inv_principle_scale= {inv_principle_scale}")
        #print(f"thinness = {self.summary.thinness}")

        self.compensator = (
            self.inflation(self.summary) * basis.T @ (rand / scale ** 0.5)
        )

        self.principle_compensator = (
            self.inflation(self.summary) * basis.T @ (1 * inv_principle_scale ** 0.5)
        )
        #print(f"radius_freq = {self.inflation(self.summary)}")
        
    def candidates(self):
        return self.summary.mean + self.compensator

    def principle_candidates(self):
        return self.summary.mean + self.principle_compensator

    def compute(self, ctx: Context) -> np.ndarray:
        values = ctx.arms @ self.candidates()

        if values.ndim == 2:
            values = np.max(values, axis=1)
        self.alpha_approx = self.summary.calculate_alpha
        self.alphas.append(self.alpha_approx)

        return values



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
        self.alphas = []

    def compute(self, ctx: Context) -> np.ndarray:
        lowers, centers, uppers = self.confidence_bounds(ctx.arms)

        # sieving arms
        baseline = lowers.max()
        optimal = uppers.max()
        minus_infty = np.ones_like(centers) * (-np.inf)

        threshold = self.tolerance * optimal + \
            (1.0 - self.tolerance) * baseline
        survivors = uppers >= threshold
        #print(f"oful_survivors = {survivors}")
        #print(f"oful_compute_uppers = {uppers}")
        #print(f"oful_compute_threshold = {threshold}")
        #print(f"oful_compute_old = {np.where(survivors, centers, lowers)}")
        #print(f"oful_compute_new = {np.where(survivors, centers, minus_infty)}")
        # computing the values
        if self.tolerance == 1.0:
            print("oful")
            self.alpha_approx = (self.summary.radius_TS()+self.summary.radius_det())/self.summary.radius_det()
            print(f"radius_TS = {self.summary.radius_TS()}, radius_det = {self.summary.radius_det()}, alpha_approx = {self.alpha_approx}")
            self.alphas.append(  self.alpha_approx          )
        else:
            self.alpha_approx = self.summary.calculate_alpha
            self.alphas.append(self.alpha_approx)



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
        #print(f"confidence_bounds = {centers - widths, centers, centers + widths}")
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


    @property
    def d(self):
        return self.summary.d
    @property
    def t(self):
        return self.summary.t


    def compute(self, ctx: Context) -> np.ndarray:
        values = ctx.arms @ self.candidates()
        true_value = ctx.arms @ self.summary.param
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
        #print(f"self.calculate_alpha = {self.calculate_alpha}, self.summary.calculate_alpha = {self.summary.calculate_alpha}")
        #print(f"SGTS,alpha_approx = {alpha_approx}")
        #print(f"SGTS,alpha = {self.alpha},alpha_approx = {alpha_approx}, self.calculate_alpha = {self.calculate_alpha}, self.summary.calculate_alpha = {self.summary.calculate_alpha},values = {values},uppers = {uppers}")
        
        if self.t == math.ceil(self.d/2*3) + 1:
            
            #print(f"exp_amount = {exp_amount},threshold = {threshold}, survivors ={survivors}")
            print(f"t = {self.t},d = {self.d}")
            #print(f"values = {values}")
            #print(f"true_value = {true_value}")
            #print(f"mean_value = {mean_value}")
            
        if alpha_approx > self.alpha:
            self.alpha_approx = alpha_approx
            self.alphas.append(self.alpha_approx)
            
            return values
        else:
            self.alpha_approx = (self.summary.radius_TS()+self.summary.radius_det())/self.summary.radius_det()
            #print("oful!")
            #print("explo!")

            #return np.where(survivors, values, minus_infty)
            self.alphas.append(self.alpha_approx)
            #print(f"oful_ratio={(self.summary.radius_TS()+self.summary.radius_det())/self.summary.radius_TS()}")
            #print(f"radius_TS = {self.summary.radius_TS()},radius_det = {self.summary.radius_det()}")

            return uppers



    def update(self):
        d = self.d
        rand = self.state.randn(d)

        basis = self.summary.basis
        scale = self.summary.scale

        inv_principle_scale  = np.zeros_like(scale)
        inv_principle_scale[0] =1/self.summary.scale[0] 
        
        self.compensator = (
            #self.inflation(self.summary) * basis.T @ (rand / scale ** 0.5)
            self.inflation(self.summary) * basis.T @ (rand / scale ** 0.5) 

        )

        self.principle_compensator = (
            self.inflation(self.summary) * basis.T @ (1 * inv_principle_scale ** 0.5)
        )
        #print(f"radius_spec = {self.inflation(self.summary)}")

        #self.beta = self.inflation(self.summary) * self.radius* np.sqrt(self.summary.lambda_) + self.noise_sd * np.sqrt(2 * np.log((self.summary.lambda_ + self.t)**(self.d/2)* self.summary.lambda_ ** ( - self.d/2)/ self.delta  ))
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



        return centers - widths, centers, centers + widths
    
    @property
    def calculate_alpha(self):
        theta_hat =  self.summary.mean
        theta_hat_Vt_norm_sqr = theta_hat.T @ self.summary.xx @ theta_hat
        lambda_max = max(self.summary.scale)
        lambda_min = min(self.summary.scale)
        
        if theta_hat_Vt_norm_sqr >= self.beta**2:
            zeta = (theta_hat_Vt_norm_sqr - self.beta**2)**0.5
            #theta_norm_lower_bound = npl.norm(theta_hat) - beta/np.sqrt()
            theta_norm_upper_bound = np.maximum( npl.norm(theta_hat - self.beta/np.sqrt(lambda_min) * self.summary.basis[self.summary.d - 1]),npl.norm(theta_hat + self.beta/np.sqrt(lambda_min) * self.summary.basis[self.summary.d - 1]))
            
            theta_norm_upper_bound = npl.norm(theta_hat) + self.beta/np.sqrt(lambda_min)
            
            x_XXnorm_max  =  np.sqrt(theta_hat.T @ self.summary.xx @ self.summary.xx @ theta_hat)/zeta
            x_XXnorm_min  =  zeta / theta_norm_upper_bound

            opt_x_XXinvnorm_sqr, wst_x_XXinvnorm_sqr = self.new_bound(self.summary.scale, np.maximum(x_XXnorm_min**2,lambda_min), np.minimum( x_XXnorm_max**2,lambda_max))
            alpha = np.sqrt(wst_x_XXinvnorm_sqr/opt_x_XXinvnorm_sqr) 
        else:
            #print("badd!")
            alpha = 0
            #opt_x_XXinvnorm_sqr, wst_x_XXinvnorm_sqr = (1/lambda_min,1/lambda_max) 
        
        return alpha



    @staticmethod
    def new_bound (scale, XX_norm_sqr_min, XX_norm_sqr_max):
        lambda_max = np.max(scale)
        lambda_min = np.min(scale)

        lambda_up = np.min(scale[scale>=XX_norm_sqr_max])
        lambda_down = np.max(scale[scale<=XX_norm_sqr_max])
        #print(f"lambda_up = {lambda_up}, lambda_down = {lambda_down}, lambda_max = {lambda_max}, lambda_min = {lambda_min}")
        
        return (1/lambda_max + 1/lambda_min - XX_norm_sqr_min/lambda_max/lambda_min), (1/lambda_up + 1/lambda_down - XX_norm_sqr_max/lambda_up/lambda_down)
