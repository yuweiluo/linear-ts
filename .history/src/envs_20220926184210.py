import abc

from typing import Optional

import numpy as np
import numpy.linalg as npl
import numpy.random as npr


class Environment:
    t: int
    param: np.ndarray  # system parameter
    ctx: Optional["Context"]
    ctx_gen: "ContextGenerator"
    noise_gen: "NoiseGenerator"

    def __init__(
        self,
        param: np.ndarray,
        ctx_gen: "ContextGenerator",
        noise_gen: "NoiseGenerator",
    ):
        self.t = -1
        self.param = param
        self.ctx = None
        self.ctx_gen = ctx_gen
        self.noise_gen = noise_gen

    def next(self):
        self.generate_context()
        self.prepare_for_context()

        return self.ctx

    def generate_context(self):
        self.t += 1
        self.ctx = self.ctx_gen.generate(self.t)

    def prepare_for_context(self):
        pass

    def get_feedback(self, arm_idx, ctx=None):
        if ctx is None:
            ctx = self.ctx
        #print(f"t-{self.t}-ctx-{ctx.arms}")
        mean = ctx.arms @ self.param
        noise = self.noise_gen.generate(ctx, arm_idx)
        max_rew = mean.max()

        return Feedback(ctx, arm_idx, mean[arm_idx], noise, max_rew)


class Context:
    def __init__(self, t, arms):
        self.t = t
        self.arms = arms


class Feedback:
    ctx: Context
    arm_idx: int

    noise: float
    max_rew: float
    mean_rew: float

    def __init__(self, ctx, arm_idx, mean_rew, noise, max_rew):
        self.ctx = ctx
        self.arm_idx = arm_idx

        self.noise = noise
        self.max_rew = max_rew
        self.mean_rew = mean_rew

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

    def __repr__(self):
        return f"LinFb(arm={self.arm_idx}, reg={self.regret}, noise={self.noise}, mean={self.mean_rew})"


class NoiseGenerator:
    def __init__(self, state=npr):
        self.state = state

    @abc.abstractmethod
    def generate(self, ctx, arm_idx):
        pass

    @staticmethod
    def gaussian_noise(sd: float, state=npr):
        class GaussianGenerator(NoiseGenerator):
            def generate(self, ctx, arm_idx):
                return self.state.randn() * sd

        return GaussianGenerator(state)


class ContextGenerator:
    @abc.abstractmethod
    def generate(self, t) -> Context:
        pass

class ConstantContextGenerator(ContextGenerator):  # class inheritance
    def __init__(self, cons_ctx):
        self.cons_ctx = cons_ctx
        
        
    def generate(self, t):
        return Context(t, self.cons_ctx)

class StochasticContextGenerator(ContextGenerator):  # class inheritance
    def __init__(self, k, d, rand_gen):
        self.k = k
        self.d = d
        self.rand_gen = rand_gen

    def generate(self, t):
        return Context(t, self.rand_gen())

    @staticmethod
    def uniform_entries(k, d, bound, state=npr):
        def _rand():
            # here the essential generator
            return state.uniform(-bound, bound, (k, d))

        return StochasticContextGenerator(k, d, _rand)

    @staticmethod
    def uniform_on_sphere(k, d, radius, state=npr):
        def _rand():
            array = state.randn(k, d)
            return radius * array / npl.norm(array, ord=2, axis=1, keepdims=True) #bug? should have axis = 1 instead of 0?

        return StochasticContextGenerator(k, d, _rand)

    @staticmethod
    def normal_entries(k, d, mean = 0.0, sd=1.0, state=npr):
        def _rand():
            return sd*state.randn(k, d)+ mean

        return StochasticContextGenerator(k, d, _rand)




class Example2ContextGenerator(ContextGenerator):
    def __init__(self, d):
        self.k = 3
        self.d = d

    def generate(self, t):
        return Context(t, self._get_context())

    def _get_context(self):
        dim = int(self.d / 3)
        x = np.zeros((self.k, self.d))
        x[0, :dim] = -1.0 / dim ** 0.5
        x[1, :dim] = -1.0 / dim ** 0.5
        x[1, dim:] = +1.0 / dim ** 0.5
        return x


class Example1ContextGenerator(ContextGenerator):
    def __init__(self, d, sign_var_mismatch):
        self.d = d
        self.sign_var_mismatch = sign_var_mismatch

    def generate(self, t):
        return Context(t, self._get_context(t))

    def _get_context(self, t):
        dim = int(self.d / 2)
        if t <= 2*dim:
            x = np.zeros((1, self.d))
            x[0, t-1] = 1
        else:
            if t <= 3*dim:
                x = np.zeros((2, self.d))
                x[0, 2 * (t - 2 * dim) - 1] = 1
                x[0, 2 * (t - 2 * dim) - 2] = 1
            else:
                x = np.zeros((2, self.d))
                x[0, :] = self.sign_var_mismatch * \
                    np.ones((1, self.d)) / (dim ** 0.5)

        return x
