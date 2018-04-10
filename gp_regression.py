import math
import torch
import gpytorch
import numpy
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

import cv2
from utils import normalize_image
# Training data
img = cv2.imread('./masked_imgs/masked_img_915_[ 0.98286092].png',0)

normalized_img = normalize_image(img)

print(normalized_img.shape)

num_row, num_col = normalized_img.shape

assert(num_row==num_col)
n = num_row

train_x = Variable(torch.from_numpy(normalized_img))
train_y = torch.zeros(int(pow(n, 2)))

for i in range(n):
	for j in range(n):
		train_y[i*n+j]=train_x.data[i,j]

train_y = Variable(train_y)
# We use KISS-GP (kernel interpolation for scalable structured Gaussian Processes)
# as in https://arxiv.org/pdf/1503.01057.pdf
class GPRegressionModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
		# Near-zero mean
		self.mean_module = ConstantMean(constant_bounds=[-1e-5, 1e-5])
		# GridInterpolationKernel over an ExactGP
		self.base_covar_module = RBFKernel(log_lengthscale_bounds=(-5, 6))
		self.covar_module = GridInterpolationKernel(self.base_covar_module, grid_size=30,
													grid_bounds=[(0, 1), (0, 1)])
		# Register the log lengthscale as a trainable parameter
		self.register_parameter('log_outputscale', nn.Parameter(torch.Tensor([0])), bounds=(-5, 6))

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		covar_x = covar_x.mul(self.log_outputscale.exp())
		return GaussianRandomVariable(mean_x, covar_x)

# Initialize the likelihood and model
# We use a Gaussian likelihood for regression so we have both a predictive mean and
# variance for our predictions
likelihood = GaussianLikelihood()
model = GPRegressionModel(train_x.data, train_y.data, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
	{'params': model.parameters()},
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

def train():
	training_iterations = 30
	for i in range(training_iterations):
		# Zero out gradients from backprop
		optimizer.zero_grad()
		# Get predictive mean and variance
		output = model(train_x)
		# Calculate loss and backprop gradients
		loss = -mll(output, train_y)
		loss.backward()
		print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.data[0]))
		optimizer.step()

train()