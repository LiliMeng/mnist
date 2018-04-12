import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import cv2
import os

from torch.autograd import Variable


import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import scipy.misc
from torch import nn, optim
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
from gpytorch.random_variables import GaussianRandomVariable

from utils import normalize_image


# Training data
def load_images_from_folder(folder):
    img_filenames = []
    labels = []
    for filename in os.listdir(folder):      
        label=filename.split('_')[2].split('.')[0]
        img_filename = os.path.join(folder,filename)
        if img_filename is not None:
            img_filenames.append(img_filename)
            labels.append(label)
    return img_filenames, labels

mask_filenames, train_mask_labels = load_images_from_folder('./masks')

n = 28

train_x = []
train_y = []
pixel_mask_counts = []
dict_pixel = {}

for i in range(len(mask_filenames)):
    img = cv2.imread(mask_filenames[i] ,0)
    mask_label = int(train_mask_labels[i])
    for i in range(n):
        for j in range(n):
            pixel_position = (i, j)
            if img[i][j] == 255:
                if pixel_position in dict_pixel:
                    dict_pixel[pixel_position] += mask_label
                else:
                    dict_pixel[pixel_position]  = mask_label
          
     
print("dict_pixel")
print(dict_pixel)
position_data = []
label_data = []

for key, value in dict_pixel.items():
    position_data.append(list(key))
    label_data.append(value)

position_data = np.asarray(position_data)
label_data = np.asarray(label_data)

# print("position_data")
# print(position_data)
pixel_x, pixel_y = position_data.T
label_data = label_data.T


result_gray_img = np.zeros((28,28,1), np.int8)
for i in range(n):
    for j in range(n):
        pixel_pos = (i,j)
        if pixel_pos in dict_pixel:
            result_gray_img[i][j] = dict_pixel[pixel_pos]


result_gray_img -= result_gray_img.min()
result_gray_img = result_gray_img/ result_gray_img.max()
result_gray_img *= 255

cv2.imwrite('./weighted_mask/weighted_mask.png', result_gray_img)


#cv2.imshow("result_img", result_gray_img)       
result_gray_img = np.array(result_gray_img, dtype = np.uint8)
result_heatmap = cv2.applyColorMap(result_gray_img, cv2.COLORMAP_JET )
cv2.imwrite('./weighted_mask/weighted_mask_heatmap.png', result_gray_img)
# cv2.imshow("result_heatmap", result_heatmap)
# cv2.waitKey()
for i in range(len(mask_filenames)):
    img = cv2.imread(mask_filenames[i] ,0)
    mask_label = int(train_mask_labels[i])
    for i in range(n):
        for j in range(n):
            # If the mask make the correct prediction, then each pixel mask has a label 0
            if mask_label == 1:
                if img[i][j] == 255:
                    train_x.append([i, j])
                    train_y.append(0)  
            # If the mask make the wrong prediciton, then each pixel mask has a label 1      
            elif mask_label == 0:
                if img[i][j] == 255:
                    train_x.append([i, j])
                    train_y.append(1) 
            else:
                raise Exception("No such labels")




train_x = Variable(torch.FloatTensor(np.asarray(train_x))).cuda()
train_y = Variable(torch.FloatTensor(np.asarray(train_y))).cuda()


print(train_x.shape)
print(train_y.shape)

print(train_x)

# # Our classification model is just KISS-GP run through a Bernoulli likelihood
# We use KISS-GP (kernel interpolation for scalable structured Gaussian Processes)
# as in https://arxiv.org/pdf/1503.01057.pdf
class GPClassificationModel(gpytorch.models.GridInducingVariationalGP):
    def __init__(self):
        super(GPClassificationModel, self).__init__(grid_size=10, grid_bounds=[(0, 28), (0, 28)])
        # Near-zero mean
        self.mean_module = ConstantMean(constant_bounds=[-1e-5, 1e-5])
        # RBF as universal approximator
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-5, 6))
        self.register_parameter('log_outputscale', nn.Parameter(torch.Tensor([0])), bounds=(-5,6))
        
    def forward(self,x):
        # Learned mean is near-zero
        mean_x = self.mean_module(x)
        # Get predictive and scale
        covar_x = self.covar_module(x)
        covar_x = covar_x.mul(self.log_outputscale.exp())
        # Store as Gaussian
        latent_pred = GaussianRandomVariable(mean_x, covar_x)
        return latent_pred

# Initialize classification model
model = GPClassificationModel().cuda()
# Likelihood is Bernoulli, warm predictive mean 
likelihood = BernoulliLikelihood().cuda()

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    # BernoulliLikelihood has no parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
# n_data refers to the amount of training data
mll = gpytorch.mlls.VariationalMarginalLogLikelihood(likelihood, model, n_data=len(train_y))

def train():
    num_training_iterations = 200
    for i in range(num_training_iterations):
        # zero back propped gradients
        optimizer.zero_grad()
       
        # Make  prediction
        output = model(train_x)
        # Calc loss and use to compute derivatives
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f' % (
            i + 1, num_training_iterations, loss.data[0],
            model.covar_module.base_kernel_module.log_lengthscale.data.squeeze()[0],
        ))
        optimizer.step()
train()

# Set model and likelihood into eval mode
model.eval()
likelihood.eval()

# Initialize figiure an axis
f, observed_ax = plt.subplots(1, 1, figsize=(4, 3))
# Test points are 100x100 grid of [0,1]x[0,1] with spacing of 1/99
n = 28
test_x = Variable(torch.zeros(int(pow(n, 2)), 2)).cuda()
for i in range(n):
    for j in range(n):
        test_x.data[i * n + j][0] = i
        test_x.data[i * n + j][1] = j
        
# Make binary predictions by warmping the model output through a Bernoulli likelihood
with gpytorch.beta_features.fast_pred_var():
    predictions = likelihood(model(test_x))

print("predictions")
print(predictions)

def ax_plot(ax, rand_var, title):
    # prob<0.5 --> label 0 // prob>0.95 --> label 1
    pred_labels = rand_var.mean().ge(0.6).float().mul(2).sub(1).cpu().data.numpy()
    # Colors = yellow for 1, red for -1
    color = []
    for i in range(len(pred_labels)):
        if pred_labels[i] == 1:
            color.append('y')
        else:
            color.append('r')
    # Plot data a scatter plot
    ax.scatter(test_x.data[:, 0].cpu().numpy(), test_x.data[:, 1].cpu().numpy(), color=color, s=1)
    ax.set_title(title)

ax_plot(observed_ax, predictions, 'Predicted Values')
plt.show()