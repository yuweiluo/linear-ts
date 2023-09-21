import argparse


import numpy as np
import pandas as pd

import torch


if torch.cuda.is_available():
    dev = "cuda:1"
else:
    dev = "cpu"
device = torch.device(dev)


class NeuralRoful:
    def choose_arm(self, ctx: Context) -> int:
    self.worth_func.bind(self.summary)  # mark
    values = self.worth_func.compute(ctx)

    return np.argmax(values).item()