import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from utils import MetricAggregator


regrets = defaultdict(MetricAggregator)
o_regrets = defaultdict(MetricAggregator)
cumregrets = defaultdict(MetricAggregator)
o_cumregrets = defaultdict(MetricAggregator)
thinnesses = defaultdict(MetricAggregator)
errors = defaultdict(MetricAggregator)
errors_candidate = defaultdict(MetricAggregator)
errors_pcandidate = defaultdict(MetricAggregator)
lambda_maxs = defaultdict(MetricAggregator)
lambda_mins = defaultdict(MetricAggregator)
lambda_maxs_over_mins = defaultdict(MetricAggregator)
worst_alphas = defaultdict(MetricAggregator)
approx_alphas = defaultdict(MetricAggregator)
discrete_alphas = defaultdict(MetricAggregator)
mus = defaultdict(MetricAggregator)
cumumus = defaultdict(MetricAggregator)
betas = defaultdict(MetricAggregator)
betas_TS = defaultdict(MetricAggregator)
zetas = defaultdict(MetricAggregator)
log_maxs_over_mins = defaultdict(MetricAggregator)
lambdas_second= defaultdict(MetricAggregator)
lambdas_third = defaultdict(MetricAggregator)
lambdas_d_minus_1 = defaultdict(MetricAggregator)
lambdas_half_d = defaultdict(MetricAggregator)
projs_first = defaultdict(MetricAggregator)

biases = defaultdict(MetricAggregator)
variances = defaultdict(MetricAggregator)
risks = defaultdict(MetricAggregator)