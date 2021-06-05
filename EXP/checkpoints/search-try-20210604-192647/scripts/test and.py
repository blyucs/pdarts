import torch
import numpy as np

reward_buffer = []
for i in range(10):
	prec1 = np.random.randn()
	reward_buffer.append(prec1)
print(np.around(torch.Tensor(reward_buffer).numpy(),3))

