import os
import torch
import pandas as pd
import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l

x = torch.arange(4.0)
print("===x===")
print(x)

x.requires_grad_(True)
print("===x.grad===")
print(x.grad)

y = 2 * torch.dot(x, x)
print("===y===")
print(y)

y.backward()
print("===x.grad===")
print(x.grad)

print("===x.grad 4*x===")
print(x.grad == 4 * x)
