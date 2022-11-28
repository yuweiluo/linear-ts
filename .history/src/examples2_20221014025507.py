import numpy as np

from envs import Environment
from envs import NoiseGenerator
from envs import StochasticContextGenerator as CtxGenerator
from envs import Example2ContextGenerator as Ex2CtxGenerator
from envs import Example1ContextGenerator as Ex1CtxGenerator
from envs import Context

from policies import Roful
from utils import timing_block


def example1_scenario(
        state_factory, k=3, d=300, t=1000, prior_var=1, prior_mu=0.0, noise_sd=1,
        thin_thresh=2.0,
        const_infl=5.0, radius = 1, delta = 0.1 
):  
    prior_var = 1
    #noise_sd = 2**0.5
    noise_sd = 5
    t = 10000

    param = state_factory().normal(prior_mu, prior_var ** 0.5, d)
    radius  = (d * prior_var) ** 0.5
    ctx_gen = Ex1CtxGenerator(d, np.sign(noise_sd - (prior_var ** 0.5)))
    noise_gen = NoiseGenerator.gaussian_noise(noise_sd, state=state_factory())

    env = Environment(param, ctx_gen, noise_gen)

    alpha = 1.0
    
    l = radius

    def radius_SG():
        return (d * np.log(1 + (l * env.t) ** 2)) ** 0.5 + (alpha * d) ** 0.5


    algs = {
        "TS-spec": Roful.spects(d, prior_var=prior_var, state=state_factory(), inflation=1.0, param=param, noise_sd = noise_sd, t = t, radius = radius, delta = 1e-4, alpha = 0.15, radius_oful=radius_SG,  ),
        "TS-Bayes": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=1.0, param=param, noise_sd = noise_sd, t = t, radius = radius, delta = 1e-4 ),
        "TS-Freq": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.radius_inflation(),
            param=param, noise_sd = noise_sd, t = t, radius = radius, delta = 1e-4
        ),
        "TS-Improved": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.conditional_inflation(
            const_infl, thin_thresh=thin_thresh),
            param=param, noise_sd = noise_sd, t = t, radius = radius, delta = 1e-4 
        ),
        'oful': Roful.oful(d, alpha=alpha, radius_SG=radius_SG, param=param, noise_sd = noise_sd, t = t, radius = radius, delta = 1e-4 ),
    }

    if k == np.inf: 
        del algs['oful']
    else:
        algs = algs
        
        #"greedy": Roful.greedy(d, prior_var=prior_var, param=param, noise_sd = noise_sd, t = t, radius = radius, delta = 0.1 ),}

    for i in range(t):

        ctx = env.next() #new context for each iteration


        for  alg_name, alg in algs.items():
            #with timing_block('choose_arm'):
            if k == np.inf:
                alg.worth_func.bind(alg.summary) 
                if alg_name == 'TS-ThinnessDirected':
                    ctx_raw = alg.worth_func.candidates().reshape(15,-1)
                else:
                    ctx_raw = alg.worth_func.candidates().reshape(1,-1)
                #print(f"ctx_raw = {ctx_raw}")
                #ctx_raw = alg.worth_func.summary.mean.reshape(1,-1)
                #ctx = Context(alg.t,ctx_raw/np.linalg.norm(ctx_raw, ord=2)*np.sqrt(3*d)) 
                #ctx = Context(alg.t,ctx_raw/np.linalg.norm(ctx_raw, ord=2)*np.sqrt(d+2)) 
                ctx = Context(alg.t,ctx_raw/np.linalg.norm(ctx_raw, ord=2)) 
                #ctx = Context(alg.t,ctx_raw/np.linalg.norm(ctx_raw, ord=2)*np.sqrt(d)) 
                #print(f"ctx_mean = {alg.worth_func.summary.mean.reshape(1,-1)}")
                #print(f"scale = {alg.worth_func.summary.scale}")
                #print(f"xx = {alg.worth_func.summary.xx}")
                #print(f"pvar = {alg.worth_func.summary.prior_var}")
                #print(f"ctx_raw = {ctx_raw}")
                #print(f"ctx = {ctx_raw}")
                #print(f"norm = {np.linalg.norm(ctx.arms,2)**2}, arms = {ctx.arms}")

            idx = alg.choose_arm(ctx)
            #print(f"iteration {i}, arm {idx}. ctx = {ctx.arms}")
            #with timing_block('get_feedback'):
            if k == np.inf:
                fb = env.get_feedback(idx, ctx) 
                #print(f"fb = {fb.chosen_arm}")
            else:
                fb = env.get_feedback(idx) 
            #with timing_block('update'):
            #if i == t-1:
                #print(f"k = {k}")
            alg.update(fb)

            '''
            if i % 100 == 0:
                print(i)
                alg.plot_hist('hist_'+str(d)+'_'+str(k)+'_'+str(i))
            '''

    return {name: alg.outputs for name, alg in algs.items()}