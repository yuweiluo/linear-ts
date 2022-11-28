import abc
from locale import normalize

from typing import List, Callable

import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt


from envs import Context
from envs import Feedback
from utils import DataSummary
from utils import predicted_risk
from utils import timing
from utils import timing_block

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import fsolve

import cvxpy as cp
import gurobipy


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

    def __init__(self, d, prior_var, worth_func: WorthFunction,  param=None, noise_sd = None, t = None, radius = None, delta = None, inflation = None):
        super().__init__()

        self.t = t
        self.radius = radius
        self.delta = delta
        self.inflation = inflation
        
        self.tt = 0
        self.thinnesses = []
        self.oracle_regrets = []
        self.errors = []
        self.errors_candidate = []
        self.errors_pcandidate = []

        self.summary = DataSummary(d, prior_var)
        self.worth_func = worth_func
        self.all_rew = 0

        self.param = param
        self.noise_sd = noise_sd
        
        self.lambda_max = []
        self.lambda_min = []
        self.lambda_max_over_min = []
        self.beta = []
        self.worst_alpha = []
        self.zeta = []
        self.approx_alpha = []


        self.lambda_second = []
        self.lambda_third = []
        self.lambda_d_minus_1 = []
        self.lambda_half_d = []
        self.proj_first = []

        self.B_n = []
        self.V_n = []
        self.R_n = []



        self.func0 = lambda x, c_0, gamma:  1/(1+ c_0 * gamma * x)
        self.func1 = lambda x, c_0, gamma:  x/(1+ c_0 * gamma * x)**2
        self.func2 = lambda x, c_0, gamma:  x**2/(1+ c_0 * gamma * x)**2


    @staticmethod
    # added param as input to calculate error
    def ts(d, prior_var, inflation=1.0, state=None, param=None, noise_sd = None, t = None, radius = None, delta = None):
        if isinstance(inflation, float):
            inflation = Roful.const_inflation(inflation)
        
            

        return Roful(d, prior_var, TsWorthFunction(inflation, state=state), param, noise_sd = noise_sd, t = t, radius = radius, delta = delta, inflation = inflation)




    @staticmethod
    # added param as input to calculate error
    def invts(d, prior_var, inflation=1.0, state=None, param=None, noise_sd = None, t = None, radius = None):
        if isinstance(inflation, float):
            inflation = Roful.const_inflation(inflation)

        return Roful(d, prior_var, InvTsWorthFunction(inflation, state=state), param, noise_sd = noise_sd, t = t, radius = radius)




    @staticmethod
    def greedy(d, prior_var, param=None, noise_sd = None, t = None, radius = None, delta = None):
        return Roful(d, prior_var, GreedyWorthFunction(), param = param, noise_sd = noise_sd, t = t, radius = radius, delta = delta)

    def pure_thin(d, prior_var, inflation=1.0, state=None):
        if isinstance(inflation, float):
            inflation = Roful.const_inflation(inflation)

        return Roful(d, prior_var, PureThinnessWorthFunction(inflation, state=state))

    def thin_dirt(d, prior_var, inflation=1.0, state=None, param=None, noise_sd = None, t = None, radius = None):
        if isinstance(inflation, float):
            inflation = Roful.const_inflation(inflation)

        return Roful(d, prior_var, ThinnessDirectedWorthFunction(inflation, state=state), param, noise_sd = noise_sd, t = t, radius = radius)

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
    def oful(d, alpha, radius_SG, param=None, noise_sd = None, t = None, radius = None, delta = None):
        return Roful(d, alpha, SievedGreedyWorthFunction(radius_SG, 1.0), param=param, noise_sd = noise_sd, t = t, radius = radius, delta = delta, inflation = Roful.const_inflation(1))




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


    #@timing
    def update_model(self, feedback: Feedback):
        x = feedback.chosen_arm
        y = feedback.rew

        self.summary.add_obs(x, y)

    #@timing
    def update_metrics(self, feedback: Feedback):
        super().update_metrics(feedback)

        
        self.tt = self.tt + 1
        
        # outputs of interest

        #with timing_block('appending'):
        self.thinnesses.append(self.summary.thinness)
        lambda_max = max(self.summary.scale)
        lambda_min = min(self.summary.scale)
        self.lambda_max.append(lambda_max)
        self.lambda_min.append(lambda_min)
        #self.lambda_second.append(self.summary.scale[1])
        #self.lambda_third.append(self.summary.scale[2])
        #self.lambda_third.append(0)
        #self.lambda_d_minus_1.append(self.summary.scale[self.d-2])
        #self.lambda_half_d.append(self.summary.scale[self.d//2])
        self.lambda_max_over_min.append(lambda_max/lambda_min)
        
        

        def calculate_alpha(beta, theta_hat, xx, scale ):
            theta_hat_Vt_norm_sqr = theta_hat.T @ xx @ theta_hat
            if theta_hat_Vt_norm_sqr >= beta**2:
                theta_norm_lower_bound = npl.norm(theta_hat) - beta/np.sqrt(lambda_min)
                theta_norm_upper_bound = (1+ beta/ np.sqrt(zeta1)) * npl.norm(theta_hat)
            opt_new, wst_new = self.new_bound(self.summary.scale, np.maximum(self.x_XXnorm_min**2,lambda_min), np.minimum( self.x_XXnorm_max_new**2,lambda_max))

        self.errors.append(np.linalg.norm(self.param - self.summary.mean)**2)
        self.oracle_regrets.append(feedback.oracle_regret) 
        #self.errors_candidate.append(np.linalg.norm(self.param - self.worth_func.candidates())**2)
        #self.errors_pcandidate.append(np.linalg.norm(self.param - self.worth_func.principle_candidates())**2)
        #print(f'delta = {self.delta}, radius = {self.radius}, noise_sd = {self.noise_sd}, t = {self.t}')
        beta = self.inflation(self.summary) * self.radius* np.sqrt(self.summary.lambda_) + self.noise_sd * np.sqrt(2 * np.log((self.summary.lambda_ + self.t)**(self.d/2)* self.summary.lambda_ ** ( - self.d/2)/ self.delta  ))
        #beta = self.radius* np.sqrt(self.summary.lambda_) + self.noise_sd * np.sqrt(2 * np.log((self.summary.lambda_ + self.t)**(self.d/2)* self.summary.lambda_ ** ( self.d/2)/ self.delta  ))



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


        theta_norm_lower_bound = npl.norm(theta_hat) - beta/np.sqrt(lambda_min)
        theta_norm_upper_bound = npl.norm(theta_hat) + beta/np.sqrt(lambda_min)
        theta_norm_upper_bound_cheat =  npl.norm(theta_hat) + beta/np.sqrt(lambda_max)

        theta_norm_upper_bound_new = (1+ beta/ np.sqrt(zeta1)) * npl.norm(theta_hat)
        theta_norm_upper_bound_new_new  = npl.norm(weight_theta_hat_upper@  self.summary.basis)
        #print(f"beta = {beta},zeta1_sqrt = {np.sqrt(zeta1)} ")
        #print(f"theta_norm_upper_bound = {theta_norm_upper_bound}, theta_norm_upper_bound_new = {theta_norm_upper_bound_new}, theta_norm_upper_bound_cheat = {theta_norm_upper_bound_cheat}")
        
        #theta_norm_interval_length =  (theta_norm_upper_bound**2 - theta_norm_lower_bound**2) /lambda_min**2

        #xx = self.summary.basis.T  * self.summary.scale @  self.summary.basis
        xx_inv = self.summary.basis.T * 1 / self.summary.scale@  self.summary.basis

        #print(f"xx - xx2 = {self.summary.xx-self.summary.basis.T  * self.summary.scale @  self.summary.basis}")

        #theta_inv_norm_interval_upper_bound = theta_hat.T @ xx_inv @ theta_hat/ npl.norm(theta_hat) **2  +  (theta_norm_upper_bound**2 - theta_norm_lower_bound**2) /lambda_min**2
        #theta_inv_norm_interval_lower_bound = theta_hat.T @ xx_inv @ theta_hat/ npl.norm(theta_hat) **2  -  (theta_norm_upper_bound**2 - theta_norm_lower_bound**2) /lambda_min**2
        
        #print(f"theta_inv_norm_interval_upper_bound = {theta_inv_norm_interval_upper_bound}, theta_inv_norm_interval_lower_bound = {theta_inv_norm_interval_lower_bound}")

        
        if zeta_square >= 0 and theta_norm_lower_bound>0:
            #print("good!")
            self.x_XXnorm_max  =  zeta / theta_norm_lower_bound 
            self.x_XXnorm_min  =  zeta / theta_norm_upper_bound_new_new
            self.x_XXnorm_max_new = np.sqrt(theta_hat.T @ self.summary.xx @ self.summary.xx @ theta_hat)/zeta
        else:
            #print("bad!")
            self.x_XXnorm_max =  np.sqrt(lambda_max)
            self.x_XXnorm_min =  np.sqrt(lambda_min)
            self.x_XXnorm_max_new = np.sqrt(lambda_max)
            

        self.x_XXinvnorm_min =  1 / np.minimum(self.x_XXnorm_max,np.sqrt(lambda_max))

        if zeta_square >= 0 and theta_norm_lower_bound>0:
            alpha = self.x_XXinvnorm_min * np.sqrt(lambda_min)
        else:
            alpha = 0


        
        #print(f"zeta = {zeta}, x_XXnorm_min = {self.x_XXnorm_min},x_XXnorm_max_new = {self.x_XXnorm_max_new}, x_XXnorm_max = {self.x_XXnorm_max}, sqrt_lambda_max = {np.sqrt(lambda_max)}")

        self.worst_alpha.append(np.sqrt(lambda_min/lambda_max))
        self.beta.append(beta)
        self.zeta.append(zeta)

        

        #self.approx_alpha.append(npl.norm(self.theta_hat)*np.sqrt(lambda_min)/np.sqrt(zeta1))
        #self.approx_alpha.append(npl.norm(self.param)*np.sqrt(lambda_min)/np.sqrt(self.param.T @ self.summary.xx @ self.param))
        #approx_alpha_  = np.sqrt(theta_inv_norm_interval_lower_bound/theta_inv_norm_interval_upper_bound)
        #self.approx_alpha.append(np.sqrt(theta_inv_norm_interval_lower_bound/theta_inv_norm_interval_upper_bound))
        #print(f"self.summary.scale = {self.summary.scale},theta_norm_lower_bound**2 = {theta_norm_lower_bound**2}, theta_norm_upper_bound**2 = {theta_norm_upper_bound**2}")
        if zeta_square >= 0 and theta_norm_lower_bound>0:
            
            #print( self.summary.scale[3])
            #opt_XX_inv_norm,opt_weight = self.solve_LP(self.summary.scale, np.maximum(self.x_XXnorm_min**2,lambda_min), np.minimum( self.x_XXnorm_max**2,lambda_max))
            #self.approx_alpha.append(self.x_XXinvnorm_min/opt_XX_inv_norm)
            #theta_opt = np.sqrt(np.array(opt_weight)) @  self.summary.basis 
            #  
            opt_new, wst_new = self.new_bound(self.summary.scale, np.maximum(self.x_XXnorm_min**2,lambda_min), np.minimum( self.x_XXnorm_max_new**2,lambda_max))
            opt_old, wst_old = self.new_bound(self.summary.scale, np.maximum(self.x_XXnorm_min**2,lambda_min), np.minimum( self.x_XXnorm_max**2,lambda_max))
            
            self.approx_alpha.append(np.sqrt(wst_new/opt_new))
            #print(f"opt_new = {opt_new}, wst_new = {wst_new}, opt_old = {opt_old}, wst_old = {wst_old}")
            #print(opt_weight)
            #self.approx_alpha.append(0)
        else:  
            #print('bad')
            opt_XX_inv_norm = 0
            opt_new, wst_new = (1/lambda_min,1/lambda_max) 
            opt_old, wst_old = (1/lambda_min,1/lambda_max)
            #theta_opt = np.zeros_like(self.param)
            self.approx_alpha.append(np.sqrt(wst_new/opt_new))
            #self.approx_alpha.append(0)

        
        #print(f"opt = {self.solve_LP(np.array([418.01738995,  60.52874083]), 64,66)})")
        
        


        #gamma = self.d/self.t 
        #print(f"self.tt= {self.tt}, zeta_square = {zeta_square}, zeta = {zeta}, zeta1 = {zeta1}, zeta2 = {zeta2}, alpha = {alpha},approx_alpha = {approx_alpha_}, lambda_min = {lambda_min}, lambda_max  = {lambda_max}")
        
        #with timing_block('solving'):
        #weight_H = np.ones(self.d) / self.d
        

        self.proj_first.append(weight_G[0])

        if self.tt == self.t:
            #print(f"basis = {self.summary.basis}")
            #print(f"scale = {self.summary.scale}, weight_G = {np.around(weight_G,decimals=2)}, sum = {np.sum(weight_G)}")
            #print(f"thinness = {self.summary.thinness}")
            #print(f"xx = {self.summary.xx}, param = {self.param}")
            #print(f"alpha = {alpha}, beta = {beta}, zeta = {zeta}, lambda_max = {lambda_max}, lambda_min = {lambda_min}, lambda_max/lambda_min = {lambda_max/lambda_min}")
            #print(f"theta_norm_lower_bound = {zeta/theta_norm_lower_bound}, theta_norm_upper_bound = {zeta/(npl.norm(self.theta_hat) + beta/np.sqrt(lambda_min))}")
            #print(f"opt_XX_inv_norm = {opt_XX_inv_norm}, x_XXinvnorm_min = {self.x_XXinvnorm_min}, x_XXinvnorm_hat = {np.sqrt(theta_hat.T @ xx_inv @ theta_hat/ npl.norm(theta_hat)**2)}")
            #print(f"opt_XX_norm! = {np.sqrt(theta_opt.T @ self.summary.xx @ theta_opt/ npl.norm(theta_opt)**2)}, self.x_XXnorm_min= {self.x_XXnorm_min}, self.x_XXnorm_max = {self.x_XXnorm_max},1/opt_XX_inv_norm = {1/opt_XX_inv_norm} ")
            #print(f"weights = {opt_weight},opt_XX_norm! = {np.sqrt(theta_opt.T @ self.summary.xx @ theta_opt/ npl.norm(theta_opt)**2)}, opt_XX_inv_norm! = {np.sqrt(theta_opt.T @ xx_inv @ theta_opt/ npl.norm(theta_opt)**2)} ")
            print(f"lambda_min = {lambda_min},  lambda_2 = {self.summary.scale[1]} ,x_XXnorm_hat**2 = {theta_hat.T @ self.summary.xx @ theta_hat/ npl.norm(theta_hat)**2}, lambda_max = {lambda_max} ")
            print(f"self.x_XXnorm_min**2 = {self.x_XXnorm_min** 2},  self.x_XXnorm_max_new**2 = {self.x_XXnorm_max_new**2}")
            #print(f"1/sqrt_scale = {1/np.sqrt(self.summary.scale)}")
            #print(f"sqrt_scale = {np.sqrt(self.summary.scale)}")
            #print(f"lambda_max = {lambda_max}, lambda_min = {lambda_min}, self.x_XXnorm_min**2 = {self.x_XXnorm_min**2}")
            #print(f"opt_new = {opt_new}, wst_new = {wst_new}, opt_XX_inv_norm**2 = {opt_XX_inv_norm**2}, self.x_XXinvnorm_min**2 = {self.x_XXinvnorm_min**2}")
            print(f"opt_new = {opt_new}, wst_new = {wst_new}, opt_old = {opt_old}, wst_old = {wst_old},x_XXinvnorm_hat**2 = {theta_hat.T @ xx_inv @ theta_hat/ npl.norm(theta_hat)**2}")
            print(f"ratio = {wst_new/opt_new}, ratio_tri = {lambda_min/lambda_max} ,opt_new = {opt_new}, 1/lambda_min = {1/lambda_min}, wst_new = {wst_new},1/lambda_max = {1/lambda_max}")
            #print(f"beta_sqr = {beta**2}, zeta_square = {zeta_square}, ")
            #print(f"npl.norm(theta_hat)**2 = {npl.norm(theta_hat)**2},   (beta/np.sqrt(lambda_min))**2 = {(beta/np.sqrt(lambda_min))**2 }, (beta/np.sqrt(lambda_max))**2 = {(beta/np.sqrt(lambda_max))**2 }")
            print(f"npl.norm(theta_hat) = {npl.norm(theta_hat)} ,  beta/np.sqrt(lambda_max) = {beta/np.sqrt(lambda_max)}, beta/np.sqrt(lambda_min) = {beta/np.sqrt(lambda_min)}")
            print(f"normalized_weight_theta_hat**2 = {np.sum(normalized_weight_theta_hat**2)}, thetahat =  {weight_theta_hat  @  self.summary.basis   },thetahat_true = {theta_hat} ")
            #print(f"xhat_t_Vtnorm = {np.sqrt(zeta1)/npl.norm(theta_hat)}, lambda_max_sqrt = {np.sqrt(lambda_max)}, lambda_min_sqrt = {np.sqrt(lambda_min)}, lambda_max/lambda_min = {lambda_max/lambda_min}")
            print(f"weight_theta_hat_upper = {weight_theta_hat_upper}, weight_theta_hat = {weight_theta_hat}")
            print(f"{theta_norm_upper_bound},{theta_norm_upper_bound_new},{theta_norm_upper_bound_new_new}, {(beta/ self.summary.scale ** 0.5  * np.sign(weight_theta_hat) )}")
            print(f"1/self.summary.scale = {1/self.summary.scale**0.5}, {weight_diff.T @ self.summary.xx @ weight_diff/d}")
            #np.array(self.summary.basis)@np.array(theta_hat) @  self.summary.basis 
            #weight_theta_hat =  ( np.array(self.summary.basis)@np.array(theta_hat))**2 / npl.norm(theta_hat)**2 
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
        
        #predicted_risk_ = [0,0,0]
        #predicted_risk_ = predicted_risk(gamma, self.radius ,self.noise_sd)
        #self.B_n.append(predicted_risk_[0])
        #self.V_n.append(predicted_risk_[1])
        #self.R_n.append(predicted_risk_[2])
        

        #print(f"t = {self.t}, B_n = {B_[0]}, V_n = {V_[0]}, R_n = {B_[0] + V_[0]}")


        

        # self.outputs = ( self.metrics.regrets, self.thinnesses, self.errors, self.lambda_max, self.lambda_min, self.lambda_second, self.lambda_third,self.B_n, self.V_n, self.R_n, self.lambda_d_minus_1, self.lambda_half_d,  self.proj_first, self.errors_candidate, self.errors_pcandidate)
        self.outputs = ( self.oracle_regrets, self.metrics.regrets, self.thinnesses, self.errors, self.lambda_max, self.lambda_min, self.lambda_max_over_min, self.proj_first, self.worst_alpha,self.approx_alpha, self.beta, self.zeta)





    @staticmethod
    def new_bound (scale, XX_norm_sqr_min, XX_norm_sqr_max):
        lambda_max = np.max(scale)
        lambda_min = np.min(scale)

        lambda_up = np.min(scale[scale>=XX_norm_sqr_max])
        lambda_down = np.max(scale[scale<=XX_norm_sqr_max])
        #print(f"lambda_up = {lambda_up}, lambda_down = {lambda_down}, lambda_max = {lambda_max}, lambda_min = {lambda_min}")
        
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
    def solve_LP2 (XX_scale, XX_norm_lower, XX_norm_upper):
        n = XX_scale.shape[0]
        x = cp.Variable(n)
        XX_scale_inv = 1/XX_scale

        constraints = [x>=0, cp.norm(x,1) == 1, x@XX_scale <=  XX_norm_upper,  x@XX_scale  >= XX_norm_lower]

        prob = cp.Problem(cp.Minimize( x@XX_scale_inv ), constraints)
        prob.solve(solver=cp.GUROBI)
        

        return np.sqrt(prob.value), x.value

    @staticmethod
    def discrete_integral( func, weight, support):
        # calulate the discrete integral of empirical distribution
        # weight: weight of each sample
        # support: support of the empirical distribution
        # func: function to be integrated
    
        # x:where to evaluate the integral
        # return: discrete integral of empirical distribution
        #print(f"  support = {support} ")
        #print(f" weight = {weight}")
        

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
        # print("lol")
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

    def candidates(self):
        return self.summary.mean + self.compensator

    def principle_candidates(self):
        return self.summary.mean + self.principle_compensator










class InvTsWorthFunction(ProductWorthFunction):
    compensator: np.ndarray

    def __init__(self, inflation, state=npr):
        self.inflation = inflation
        self.state = state

    def compute(self, ctx: Context) -> np.ndarray:
        values = 1/ctx.arms @ self.candidates()

        if values.ndim == 2:
            values = np.max(values, axis=1)

        return values

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

class SgTsWorthFunction(ProductWorthFunction):
    compensator: np.ndarray

    def __init__(self, inflation, state=npr):
        self.inflation = inflation
        self.state = state

    def compute(self, ctx: Context) -> np.ndarray:
        values = ctx.arms @ self.candidates()
        lowers, centers, uppers = self.confidence_bounds(ctx.arms)

        

        if values.ndim == 2:
            values = np.max(values, axis=1)


        

        return values

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
        
        self.compensator = (
            self.inflation(self.summary) * basis.T @ (rand / scale ** 0.5)
        )

        self.principle_compensator = (
            self.inflation(self.summary) * basis.T @ (1 * inv_principle_scale ** 0.5)
        )

    def candidates(self):
        return self.summary.mean + self.compensator

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