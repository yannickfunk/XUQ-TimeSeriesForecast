import matplotlib.pyplot as plt
import numpy as np
import torch

torch.cuda.is_available()

# draw a sine curve and show the plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()

a: int = 2
