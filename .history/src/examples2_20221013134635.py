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