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
        state_factory, k=3, d=300, t=1000, prior_var=1, prior_mu=0.0, noise_sd=10.0,
        thin_thresh=2.0,
        const_infl=5.0
):
    param = state_factory().normal(prior_mu, prior_var ** 0.5, d)
    ctx_gen = Ex1CtxGenerator(d, np.sign(noise_sd - (prior_var ** 0.5)))
    noise_gen = NoiseGenerator.gaussian_noise(noise_sd, state=state_factory())

    env = Environment(param, ctx_gen, noise_gen)

    algs = {
        "TS-Bayes": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=1.0),
        # "OFUL": Roful.oful(
        #     d,
        #     alpha = 1,
        #     radius=Roful.radius_inflation()),
        # "TS-2": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=5.0),
        "TS-Freq": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.radius_inflation(),
        ),
        "TS-Improved": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.conditional_inflation(
                const_infl, thin_thresh=thin_thresh),
        ),
        "TS-Thinness": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.dynamic_inflation(),
        ),

        "TS-ThinnessDirected": Roful.thin_dirt(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=1.0,
        ),

    }

    for i in range(t):
        ctx = env.next()

        for alg in algs.values():
            idx = alg.choose_arm(ctx)
            fb = env.get_feedback(idx)
            alg.update(fb)

        if i % 100 == 0:
            print(i)

    return {name: (alg.metrics.regrets, alg.thinnesses) for name, alg in algs.items()}


def example2_scenario(
        state_factory, k=3, d=300, t=1000, prior_var=1.0, prior_mu=10.0, noise_sd=1.0,
        thin_thresh=2.0,
        const_infl=5.0
):
    param = state_factory().normal(prior_mu, prior_var ** 0.5, d)
    if k != 3:
        print('*** Warning ***, k should be 3 for fixed action case')
    ctx_gen = Ex2CtxGenerator(d)
    noise_gen = NoiseGenerator.gaussian_noise(noise_sd, state=state_factory())

    env = Environment(param, ctx_gen, noise_gen)

    algs = {
        "TS-Bayes": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=1.0),
        # "OFUL": Roful.oful(
        #    d,
        #    alpha = 1,
        #    radius=Roful.radius_inflation()),
        # "TS-2": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=5.0),
        "TS-Freq": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.radius_inflation(),
        ),
        "TS-Improved": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.conditional_inflation(
                const_infl, thin_thresh=thin_thresh),
        ),
        "TS-Thinness": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.dynamic_inflation(),
        ),


        "TS-ThinnessDirected": Roful.thin_dirt(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=1.0,
        ),

    }

    for i in range(t):
        ctx = env.next()

        for alg in algs.values():
            idx = alg.choose_arm(ctx)
            fb = env.get_feedback(idx)
            alg.update(fb)

        if i % 100 == 0:
            print(i)

    return {name: (alg.metrics.regrets, alg.thinnesses) for name, alg in algs.items()}


def russo_scenario(state_factory, k=100, d=100, t=1000, prior_var=10.0, arm_bound=0.1, prior_mu=0.0, noise_sd=1.0, thin_thresh=2.0, const_infl=5.0):

    param = state_factory().normal(prior_mu, prior_var**0.5, d)
    ctx_gen = CtxGenerator.uniform_entries(k, d, arm_bound / d**0.5, state=state_factory())

    noise_gen = NoiseGenerator.gaussian_noise(noise_sd, state=state_factory())
    env = Environment(param, ctx_gen, noise_gen)
    algs = {"TS-Bayes": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=1.0, param=param, noise_sd = noise_sd, t = t),}

    for i in range(t):
        ctx = env.next()
        for alg in algs.values():
            idx = alg.choose_arm(ctx)
            fb = env.get_feedback(idx)
            alg.update(fb)
            '''
            if i % 100 == 0:
                print(i)
                alg.plot_hist('hist_'+str(d)+'_'+str(k)+'_'+str(i))
            '''

    return {name: alg.outputs for name, alg in algs.items()}


def russo_scenario2(state_factory, k=1, d=100, t=1000, prior_var=2.0, arm_bound=0.1, prior_mu=0.0, noise_sd=1.0, thin_thresh=2.0, const_infl=5.0, radius=1.0):

    param = state_factory().normal(prior_mu, prior_var**0.5, d)
    param = radius * param / np.linalg.norm(param, ord=2)
    #ctx_gen = CtxGenerator.uniform_entries(k, d, arm_bound / d**0.5, state=state_factory())
    #ctx_gen = CtxGenerator.uniform_entries(k, d, np.sqrt(3), state=state_factory())
    if k == np.inf:
        print(f"using special ctx generator for k = {k}")
        #ctx_gen = CtxGenerator.normal_entries(1, d, mean = 0, sd = 1, state=state_factory())
        #ctx_gen = CtxGenerator.uniform_entries(1, d, np.sqrt(3), state=state_factory())
        #ctx_gen = CtxGenerator.uniform_in_ball(1, d,  np.sqrt(d+2), state=state_factory())
        ctx_gen = CtxGenerator.uniform_on_sphere(1, d,  np.sqrt(d), state=state_factory())
    else:
        k = int(k)
        #ctx_gen = CtxGenerator.normal_entries(k, d, mean = 0, sd = 1, state=state_factory())
        #ctx_gen = CtxGenerator.uniform_entries(k, d, np.sqrt(3), state=state_factory())
        #ctx_gen = CtxGenerator.uniform_in_ball(k, d,  np.sqrt(d+2), state=state_factory()) #variance of d-unit ball is 1/(d+2)
        ctx_gen = CtxGenerator.uniform_on_sphere(k, d,  np.sqrt(d), state=state_factory())

    #ctx_gen = CtxGenerator.uniform_on_sphere(k, d, radius, state=state_factory())
    #ctx_gen = CtxGenerator.uniform_on_sphere(k, d, np.sqrt(d), state=state_factory())

    noise_gen = NoiseGenerator.gaussian_noise(noise_sd, state=state_factory())
    env = Environment(param, ctx_gen, noise_gen)

    algs = {"TS-Bayes": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=1.0, param=param, noise_sd = noise_sd, t = t, radius = radius ),"greedy": Roful.greedy(d, prior_var=prior_var, param=param, noise_sd = noise_sd, t = t, radius = radius ),}

    for i in range(t):

        ctx = env.next() #new context for each iteration


        for alg in algs.values():
            #with timing_block('choose_arm'):
            if k == np.inf:
                alg.worth_func.bind(alg.summary) 
                ctx_raw = alg.worth_func.candidates().reshape(1,-1)
                #ctx_raw = alg.worth_func.summary.mean.reshape(1,-1)
                #ctx = Context(alg.t,ctx_raw/np.linalg.norm(ctx_raw, ord=2)*np.sqrt(3*d)) 
                #ctx = Context(alg.t,ctx_raw/np.linalg.norm(ctx_raw, ord=2)*np.sqrt(d+2)) 
                ctx = Context(alg.t,ctx_raw/np.linalg.norm(ctx_raw, ord=2)*np.sqrt(d)) 
                print(f"ctx_mean = {alg.worth_func.summary.mean.reshape(1,-1)}")
                print(f"scale = {alg.worth_func.summary.scale}")
                print(f"xx = {alg.worth_func.summary.xx}")
                print(f"pvar = {alg.worth_func.summary.prior_var}")
                print(f"ctx_raw = {ctx_raw}")
                print(f"ctx = {ctx_raw}")
                print(f"norm = {np.linalg.norm(ctx.arms,2)**2}, arms = {ctx.arms}")

            idx = alg.choose_arm(ctx)
            #print(f"iteration {i}, arm {idx}. ctx = {ctx.arms}")
            #with timing_block('get_feedback'):
            if k == np.inf:
                fb = env.get_feedback(idx, ctx) 
                print(f"fb = {fb.chosen_arm}")
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


def russo_scenario3(state_factory, k=1, d=100, t=1000, prior_var=2.0, arm_bound=0.1, prior_mu=0.0, noise_sd=1.0, thin_thresh=2.0, const_infl=5.0, radius=1.0):

    param = state_factory().normal(prior_mu, prior_var**0.5, d)
    param = radius * param / np.linalg.norm(param, ord=2)

    #ctx_gen = CtxGenerator.uniform_entries(k, d, arm_bound / d**0.5, state=state_factory())
    #ctx_gen = CtxGenerator.uniform_entries(k, d, np.sqrt(3), state=state_factory())
    if k in [0, np.inf]:
        print(f"using special ctx generator for k = {k}")
        #ctx_gen = CtxGenerator.normal_entries(1, d, mean = 0, sd = 1, state=state_factory())
        ctx_gen = CtxGenerator.uniform_entries(1, d, np.sqrt(3), state=state_factory())
    else:
        k = int(k)
        #ctx_gen = CtxGenerator.normal_entries(k, d, mean = 0, sd = 1, state=state_factory())
        ctx_gen = CtxGenerator.uniform_entries(k, d, np.sqrt(3), state=state_factory())

    #ctx_gen = CtxGenerator.uniform_on_sphere(k, d, radius, state=state_factory())
    #ctx_gen = CtxGenerator.uniform_on_sphere(k, d, np.sqrt(d), state=state_factory())

    noise_gen = NoiseGenerator.gaussian_noise(noise_sd, state=state_factory())
    env = Environment(param, ctx_gen, noise_gen)

    if k == 0:
        algs = {"TS-Bayes": Roful.greedy(d, prior_var=prior_var, param=param, noise_sd = noise_sd, t = t, radius = radius ),        }
    else:    
        algs = {"TS-Bayes": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=1.0, param=param, noise_sd = noise_sd, t = t, radius = radius ),        
        "TS-Improved": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.conditional_inflation(
                const_infl, thin_thresh=thin_thresh),
            param=param,noise_sd = noise_sd, t = t, radius = radius 
        ),
        "TS-ThinnessDirected": Roful.thin_dirt(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=1.0,
            param=param,noise_sd = noise_sd, t = t, radius = radius 
        ),}

    for i in range(t):

        ctx = env.next() #new context for each iteration


        for alg in algs.values():
            #with timing_block('choose_arm'):
            if k == np.inf:
                alg.worth_func.bind(alg.summary) 
                ctx_raw = alg.worth_func.candidates().reshape(1,-1)
                ctx = Context(alg.t,ctx_raw/np.linalg.norm(ctx_raw, ord=2)*np.sqrt(3*d)) 

            idx = alg.choose_arm(ctx)
            #print(f"iteration {i}, arm {idx}. ctx = {ctx.arms}")
            #with timing_block('get_feedback'):
            if k == np.inf:
                fb = env.get_feedback(idx, ctx) 
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


def russo_scenario_all(
    state_factory, k=100, d=100, t=1000, prior_var=10.0, arm_bound=0.1,
        prior_mu=0.0, noise_sd=1.0,
        thin_thresh=2.0,
        const_infl=5.0
):
    param = state_factory().normal(prior_mu, prior_var ** 0.5, d)
    ctx_gen = CtxGenerator.uniform_entries(
        k, d, arm_bound / d ** 0.5, state=state_factory())
    noise_gen = NoiseGenerator.gaussian_noise(noise_sd, state=state_factory())

    env = Environment(param, ctx_gen, noise_gen)
    # print('param')
    # print(param)
    algs = {
        "TS-Bayes": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=1.0, param=param),
        # "TS-2": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=5.0),
        # "OFUL": Roful.oful(
        #    d,
        #    alpha=1,
        #    radius=Roful.radius_inflation()),

        "TS-Freq": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.radius_inflation(),
            param=param
        ),
        "TS-Improved": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.conditional_inflation(
                const_infl, thin_thresh=thin_thresh),
            param=param
        ),
        "TS-Thinness": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.dynamic_inflation(),
            param=param
        ),

        "TS-ThinnessDirected": Roful.thin_dirt(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=1.0,
            param=param
        ),
    }

    for i in range(t):
        ctx = env.next()

        for alg in algs.values():
            idx = alg.choose_arm(ctx)
            fb = env.get_feedback(idx)
            alg.update(fb)

            if i % 100 == 0:
                print(i)
                alg.plot_hist(f'hist_{str(i)}')

    return {name: (alg.metrics.regrets, alg.thinnesses, alg.errors, alg.lambda_max, alg.lambda_min) for name, alg in algs.items()}
