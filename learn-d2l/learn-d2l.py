import os
import torch
import time
import pandas as pd
import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l

print("********learn d2l********")

# use a GPU if available, otherwise use the CPU
device = d2l.try_gpu()
# test the device
print(device)


