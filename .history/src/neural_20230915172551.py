import argparse


import numpy as np
import pandas as pd

import torch


if torch.cuda.is_available():
    dev = "cuda:1"
else:
    dev = "cpu"
device = torch.device(dev)
