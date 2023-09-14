import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import pandas as pd
import time
from contextlib import contextmanager
from statsmodels.distributions.empirical_distribution import ECDF

def timing(func):
    def outer(*args, **kwargs):
        start_time = time.time()
        inner = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"func @{func.__name__} took {1000 * elapsed_time} ms")
        #print('{:s} function took {:.3f} ms'.format(func.__name__, (time2-time1)*1000.0))

        return inner
    return outer

@contextmanager
def timing_block(name):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print(f"context #{name} took {1000 * elapsed_time} ms")


class DataSummary:
    lambda_: float

    xy: np.ndarray
    xx: np.ndarray

    _mean: np.ndarray
    _basis: np.ndarray
    _scale: np.ndarray
    _dirty: bool

    def __init__(self, dim, lambda_, noise_sd, param_bound, param, inflation,T):
        self.lambda_ = lambda_
        # self.param_bound = (dim * self.prior_var) ** 0.5  # FIXME
        self.param_bound = param_bound
        self.noise_sd = noise_sd
    
        self.xy = np.zeros(dim, dtype=np.float)
        #self.xx = np.eye(dim, dtype=np.float) *lambda_ * noise_sd**2 # FIXME 
        self.xx = np.eye(dim, dtype=np.float) *lambda_ 

        self._mean = np.zeros(dim, dtype=np.float)
        self._basis = np.eye(dim, dtype=np.float)
        self._scale = np.ones(dim, dtype=np.float) *lambda_
        self._dirty = False
        self.t = 0
        self.param = param
        self.inflation = inflation
        self.T = T
        



    #@timing
    def _update_caches(self):
        # with timing_block("svd"):
        # the most time consuming part

        
        svd = npl.svd(self.xx, hermitian=True)

        self._mean = svd[0] @ ((svd[2] @ self.xy) / svd[1])
        self._basis = svd[2]
        self._scale = svd[1]
        #self.betas.append(self.radius_det())
        self._dirty = False

    #def add_obs(self, x, y, tau=1.0):
    #    self.xy += x * y / tau ** 2
    #    self.xx += np.outer(x, x) / tau ** 2
    #    self.t += 1

    #    self._dirty = True
    
    def add_obs(self, x, y, tau=1.0):
        self.xy += x * y / tau ** 2
        self.xx += np.outer(x, x) / tau ** 2
        self.t += 1

        self._dirty = True
        
    @property
    def d(self):
        return self.xy.shape[0]

    @property
    def TS_prior_var(self):
        return 1.0 / self.lambda_


    @property
    def mean(self) -> np.ndarray:
        if self._dirty:
            self._update_caches()

        return self._mean

    @property
    def basis(self) -> np.ndarray:
        if self._dirty:
            self._update_caches()

        return self._basis

    @property
    def scale(self) -> np.ndarray:
        if self._dirty:
            self._update_caches()

        return self._scale

    @property
    def thinness(self):
        if self._dirty:
            self._update_caches()
        scale_inv = 1 / self.scale
        return (max(scale_inv) * self.d / sum(scale_inv)) ** 0.5


    def radius_det(self, delta=1e-3):
        if self._dirty:
            self._update_caches()
        term1 = np.log(self.scale / self.lambda_).sum() - 2 * np.log(delta/2/self.T)
        term2 = self.lambda_ * self.param_bound ** 2
        #print(f"self.T = {self.T}, delta = {delta},radius_det = {self.noise_sd * term1 ** 0.5 + term2 ** 0.5}")
        return self.noise_sd * term1 ** 0.5 + term2 ** 0.5


    def radius_OFUL(self, delta=1e-3):
        if self._dirty:
            self._update_caches()
        return self.radius_det(delta=delta)

    def radius_TS(self, delta=1e-3 ):
        if self._dirty:
            self._update_caches()
        return self.inflation(self)*self.radius_det(delta = delta)
    

    def radius_normal(self, delta=1e-3 ):

        if self._dirty:
            self._update_caches()

        return np.sqrt(2*self.d*np.log(2*self.d/delta*2*self.T))
    
    

    @staticmethod
    def new_bound (scale, XX_norm_sqr_min, XX_norm_sqr_max):
        lambda_max = np.max(scale)
        lambda_min = np.min(scale)

        lambda_up = np.min(scale[scale>=XX_norm_sqr_max])
        lambda_down = np.max(scale[scale<=XX_norm_sqr_max])
        #print(f"lambda_up = {lambda_up}, lambda_down = {lambda_down}, lambda_max = {lambda_max}, lambda_min = {lambda_min}")
        
        return (1/lambda_max + 1/lambda_min - XX_norm_sqr_min/lambda_max/lambda_min), (1/lambda_up + 1/lambda_down - XX_norm_sqr_max/lambda_up/lambda_down)

    @property
    def calculate_alpha(self):
        #print(f"radius_TS()= {self.radius_TS()}, radius_det() = {self.radius_det()}")
        ww, wst_x_XXinvnorm_sqr_TS = self.calculate_x_XXinvnorm_sqr_range( self.radius_TS())
        opt_x_XXinvnorm_sqr_RLS, oo = self.calculate_x_XXinvnorm_sqr_range( self.radius_det())
        #print(f"ww = {ww}, oo = {oo}")
        #print(f"wst_x_XXinvnorm_sqr_TS = {wst_x_XXinvnorm_sqr_TS}, opt_x_XXinvnorm_sqr_RLS = {opt_x_XXinvnorm_sqr_RLS}")
        #print(f"new_alpha = {np.sqrt(oo/opt_x_XXinvnorm_sqr_RLS)}")
        #print(f"old_alpha = {np.sqrt(wst_x_XXinvnorm_sqr_TS/opt_x_XXinvnorm_sqr_RLS)}")
        alpha = np.sqrt(opt_x_XXinvnorm_sqr_RLS/wst_x_XXinvnorm_sqr_TS) 
        #print(f"alpha = {alpha}")

        return alpha
    @property
    def calculate_mu(self):
        #print(f"calculate_mu = {(self.inflation(self)+1.0) / self.calculate_alpha} ")
        return (self.inflation(self)+1.0) * (1 + self.calculate_alpha)

    def calculate_x_XXinvnorm_sqr_range(self,  ellipsoid_radius = None):
        if self._dirty:
            self._update_caches()
        theta_hat =  self.mean
        #print(f"theta_hat = {theta_hat}")
        theta_hat_Vt_norm_sqr = theta_hat.T @ self.xx @ theta_hat
        lambda_max = max(self.scale)
        lambda_min = min(self.scale)
        
        if ellipsoid_radius == 0.0:
            theta_hat_Vt_inv_norm_sqr =theta_hat.T @ npl.solve(self.xx, theta_hat)
            #print(f"theta_hat_Vt_inv_norm_sqrr = {theta_hat_Vt_inv_norm_sqr}")
            if theta_hat_Vt_inv_norm_sqr == 0.0:
                return 0.0, 0.0
            else:
                return theta_hat_Vt_inv_norm_sqr /npl.norm(theta_hat)**2, theta_hat_Vt_inv_norm_sqr /npl.norm(theta_hat)**2
        
        
        else: 
            if theta_hat_Vt_norm_sqr >= ellipsoid_radius**2:
            
                zeta = (theta_hat_Vt_norm_sqr - ellipsoid_radius**2)**0.5
                #theta_norm_lower_bound = npl.norm(theta_hat) - beta/np.sqrt()
                theta_norm_upper_bound = np.maximum( npl.norm(theta_hat - ellipsoid_radius/np.sqrt(lambda_min) * self.basis[self.d - 1]),npl.norm(theta_hat + ellipsoid_radius/np.sqrt(lambda_min) * self.basis[self.d - 1]))
                theta_norm_upper_bound = npl.norm(theta_hat) + ellipsoid_radius/np.sqrt(lambda_min)

                
                x_XXnorm_max  =  np.sqrt(theta_hat.T @ self.xx @ self.xx @ theta_hat)/zeta
                x_XXnorm_min  =  zeta / theta_norm_upper_bound

                opt_x_XXinvnorm_sqr, wst_x_XXinvnorm_sqr = self.new_bound(self.scale, np.maximum(x_XXnorm_min**2,lambda_min), np.minimum( x_XXnorm_max**2,lambda_max))
            else:
                #opt_x_XXinvnorm_sqr = 1/lambda_min
                #wst_x_XXinvnorm_sqr = 0
                
                opt_x_XXinvnorm_sqr = 1/lambda_min
                wst_x_XXinvnorm_sqr = 1/lambda_max
            return opt_x_XXinvnorm_sqr, wst_x_XXinvnorm_sqr

    def confidence_center(self, arms):
        #print(f"confidence_center = {arms @ self.summary.mean}")

        return arms @ self.mean
        
    def confidence_width(self, arms,ellipise_radius):
        scale = arms @ npl.solve(self.xx, arms.T)
        if len(scale.shape) == 2:
            scale = np.diag(scale)

        return ellipise_radius* scale ** 0.5, scale
    
    def confidence_bounds(self, arms, ellipise_radius):
        centers = self.confidence_center(arms)
        widths,_ = self.confidence_width(arms,ellipise_radius)
        return centers - widths, centers, centers + widths


    def calculate_discrete_alpha(self, arms):
        lowers_CA, centers_CA, uppers_CA = self.confidence_bounds(arms, self.radius_TS())
        lowers_RLS, centers_RLS, uppers_RLS = self.confidence_bounds(arms, self.radius_det())

        survivors_CA = uppers_CA >= np.max(lowers_CA)
        survivors_RLS = uppers_RLS >= np.max(lowers_RLS)

        _, x_XXinvnorm_sqr = self.confidence_width( arms,self.radius_det())
        
        
        
        nome = np.sqrt(np.max(x_XXinvnorm_sqr[survivors_RLS]))
        deno = np.sqrt(np.min(x_XXinvnorm_sqr[survivors_CA]))
        #print(f"nome = {nome},deno = {deno}")
        alpha = nome/deno
        #print(f"alpha = {alpha}")

        return alpha



class MetricAggregator:
    def __init__(self):
        self.m0 = []
        self.m1 = []
        self.m2 = []

    def confidence_band(self):
        m0 = np.array(self.m0)
        m1 = np.array(self.m1)
        m2 = np.array(self.m2)

        m0 = np.maximum(m0, 1)

        mean = m1 / m0
        var = (m2 - m1 ** 2 / m0) / (m0 - 1)
        sd = var ** 0.5
        se = (var / m0) ** 0.5

        return mean, sd, se

    def plot(self, ax, label,marker,mark_num, scale=2.0):
        mean, sd, se = self.confidence_band()

        x = np.arange(len(mean))

        lower = mean - scale * se
        upper = mean + scale * se

        ax.fill_between(x, lower, upper, alpha=0.2)
        markevery = int(len(x)/mark_num)
        ax.plot(x, mean, label=label, marker=marker, markevery=markevery, markersize=5)

    def get_mean_se(self):
        mean, sd, se = self.confidence_band()

        return mean, se

    def aggregate(self, xs, filter=lambda _: True):
        self._ensure_len(len(xs))

        for i, x in enumerate(xs):
            if filter(i):
                self.m0[i] += 1
                self.m1[i] += x
                self.m2[i] += x ** 2

    def _ensure_len(self, n):
        dn = n - len(self.m0)

        if dn > 0:
            self.m0 += [0] * dn
            self.m1 += [0] * dn
            self.m2 += [0] * dn


class StateFactory:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self):
        state = npr.RandomState(self.seed)

        self.seed += 1

        return state


def sample_ecdf(qe,pe = None,u=None,data_range=[0,1]):
    #qe: empirical quantile 
    #pe: empirical cumulative probability
    #u: uniform random variable, if specified then return inverse cdf value
    #data_range: range of data

    if u is None:
        u = np.random.uniform(0, 1)

    if pe is None:
        pe = np.linspace(0, 1, len(qe))
    if len(qe) != len(pe):
        raise ValueError('length of qe and pe must be the same')

    qe_array = np.array(qe)
    pe_array = np.array(pe)

    result = (np.min(qe_array[pe_array>=u]) if u <= max(pe) else data_range[1]) 

    return result

def predicted_risk(gamma, radius, noise_sd): 
    #limiting risk of the min-norm ridge estimator for isotropic features 
    #gamma: ratio
    #radius: radius of the coefficient
    #noise_sd: standard deviation of the noise

    if gamma == 1:
        raise ValueError('gamma cannot be 1') 
    if gamma < 1:
        bias_ = 0
        variance_ = noise_sd**2 * gamma/(1-gamma) 
    else:
        bias_ = radius**2 *(1 - 1/gamma)
        variance_ = noise_sd**2 * 1 /(gamma - 1)   

    return  [bias_, variance_, bias_+variance_]

def discrete_integral( func, weight, support):
    # calulate the discrete integral of empirical distribution
    # weight: weight of each sample
    # support: support of the empirical distribution
    # func: function to be integrated

    # x:where to evaluate the integral
    # return: discrete integral of empirical distribution

    return np.sum(func(support) @ weight)




