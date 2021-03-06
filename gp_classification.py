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

dataset = "MNIST"

if dataset == "MNIST":
    n = 28
else:
    raise Exception("Not implemented yet")

mode = 'Train'

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


def prepare_training_data():
    mask_filenames, train_mask_labels = load_images_from_folder('./masks')

    train_x = []
    train_y = []
    pixel_mask_counts = []
    dict_pixel = {}

    for i in range(len(mask_filenames)):
        img = cv2.imread(mask_filenames[i] ,0)
        mask_label = int(train_mask_labels[i])
       
        for j in range(n):
            for k in range(n):
                pixel_position = (j, k)
                if img[j][k] == 0:
                    if pixel_position in dict_pixel:
                        dict_pixel[pixel_position] += mask_label
                    else:
                        dict_pixel[pixel_position]  = mask_label
              
         


    result_gray_img = np.zeros((n,n,1), np.int8)
    for i in range(n):
        for j in range(n):
            pixel_pos = (i,j)
            if pixel_pos in dict_pixel:
                result_gray_img[i][j] = dict_pixel[pixel_pos]


    result_gray_img -= result_gray_img.min()
    result_gray_img = result_gray_img/ result_gray_img.max()
    result_gray_img *= 255

    cv2.imwrite('./weighted_mask/weighted_mask.png', result_gray_img)

     
    result_gray_img = np.array(result_gray_img, dtype = np.uint8)
    result_heatmap = cv2.applyColorMap(result_gray_img, cv2.COLORMAP_JET )

    cv2.imwrite('./weighted_mask/weighted_mask_heatmap.png', result_heatmap)
    cv2.imshow("result_heatmap", result_heatmap)
    cv2.waitKey()
    cv2.destroyAllWindows()

    for i in range(len(mask_filenames)):
        img = cv2.imread(mask_filenames[i] ,0)
        mask_label = int(train_mask_labels[i])
        for j in range(n):
            for k in range(n):
                # If the mask make the correct prediction, then these pixels can be masked, each pixel mask has a label 0
                if mask_label == 1:
                    if img[j][k] == 0:
                        train_x.append([j, k])
                        train_y.append(0)  
                # If the mask make the wrong prediciton, then these pixels cannot be masked, then each pixel mask has a label 1      
                elif mask_label == 0:
                    if img[j][k] == 0:
                        train_x.append([j, k])
                        train_y.append(1) 
                else:
                    raise Exception("No such labels")



    train_x = Variable(torch.FloatTensor(np.asarray(train_x))).cuda()
    train_y = Variable(torch.FloatTensor(np.asarray(train_y))).cuda()
    
    print("train_x.shape: ", train_x.shape)
    print("train_y.shape: ", train_y.shape)

    return train_x, train_y

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


def train(train_x, train_y, model, likelihood):

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

    num_training_iterations = 10
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

    torch.save(model.state_dict(), './gp_saved_checkpoints/gp_cls_checkpoint.pth.tar')


def eval_superpixels(model, likelihood):

    model.load_state_dict(torch.load('./gp_saved_checkpoints/gp_cls_checkpoint.pth.tar'))
    # Set model and likelihood into eval mode
    model.eval()
    likelihood.eval()

    # Initialize figiure an axis
    f, observed_ax = plt.subplots(1, 1, figsize=(4, 3))
    # Test points are 100x100 grid of [0,1]x[0,1] with spacing of 1/99

    test_x = Variable(torch.zeros(int(pow(n, 2)), 2)).cuda()
    for i in range(n):
        for j in range(n):
            test_x.data[i * n + j][0] = i
            test_x.data[i * n + j][1] = j
            
    # Make binary predictions by warmping the model output through a Bernoulli likelihood
    with gpytorch.beta_features.fast_pred_var():
        predictions = likelihood(model(test_x))

    final_predictions = predictions.mean().cpu().data.numpy()

    return final_predictions

def plot_result(final_predictions):

    mask_filenames, train_mask_labels = load_images_from_folder('./masks')

    train_x = []
    train_y = []
    pixel_mask_counts = []
    dict_pixel = {}

    for i in range(len(mask_filenames)):
        img = cv2.imread(mask_filenames[i] ,0)
        mask_label = int(train_mask_labels[i])
       
        for j in range(n):
            for k in range(n):
                pixel_position = (j, k)
                if img[j][k] == 0:
                    if pixel_position in dict_pixel:
                        dict_pixel[pixel_position] += mask_label
                    else:
                        dict_pixel[pixel_position]  = mask_label
              
    result_gray_img = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            pixel_pos = (i,j)
            if pixel_pos in dict_pixel:
                result_gray_img[i][j] = dict_pixel[pixel_pos]


    result_gray_img -= result_gray_img.min()
    result_gray_img = result_gray_img/ result_gray_img.max()
    result_gray_img *= 255
    
    result_gray_img = np.array(result_gray_img, dtype = np.uint8)
    result_heatmap = cv2.applyColorMap(result_gray_img, cv2.COLORMAP_JET )

    org_test_gray_img = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            org_test_gray_img[i][j] = final_predictions[i*n+j]
           

    test_gray_img = org_test_gray_img
    test_gray_img -= test_gray_img.min()
    test_gray_img = test_gray_img/ test_gray_img.max()
    test_gray_img *= 255


    test_gray_img = np.array(test_gray_img, dtype = np.uint8)

    cv2.imwrite('./weighted_mask/pred_mask.png', test_gray_img)

    test_heatmap = cv2.applyColorMap(test_gray_img, cv2.COLORMAP_JET )
    cv2.imwrite('./weighted_mask/pred_mask_heatmap.png', test_heatmap)



    org_img = cv2.imread('original_img_index2_label_9.png', 0)


    final_masked_img = org_img * org_test_gray_img 
    #final_masked_img = final_masked_img.transpose(1, 2, 0)
    final_masked_img -= final_masked_img.min()
    final_masked_img /= final_masked_img.max()
    final_masked_img *= 255
    final_masked_img = np.array(final_masked_img, dtype = np.uint8)
    final_masked_img_color = cv2.cvtColor(final_masked_img, cv2.COLOR_GRAY2RGB)

    org_img_3channel = cv2.imread('original_img_index2_label_9.png')

    plt.subplot(141),plt.imshow(org_img_3channel,'gray'),plt.title('Original img')
    plt.subplot(142),plt.imshow(cv2.cvtColor(result_heatmap, cv2.COLOR_BGR2RGB),'jet'),plt.title('Summed label training heatmap')
    plt.subplot(143),plt.imshow(cv2.cvtColor(test_heatmap, cv2.COLOR_BGR2RGB),'gray'),plt.title('Predicted mask heatmap')
    plt.subplot(144),plt.imshow(cv2.cvtColor(final_masked_img_color, cv2.COLOR_BGR2RGB),'gray'),plt.title('Org_img with predicted mask')

    plt.colorbar()
    #plt.set_cmap('Reds')
    plt.set_cmap('seismic')
    plt.show()


def main():

    # Initialize classification model
    model = GPClassificationModel().cuda()
    # Likelihood is Bernoulli, warm predictive mean 
    likelihood = BernoulliLikelihood().cuda()  

    if mode == 'Train':
        train_x, train_y = prepare_training_data()

    
        train(train_x, train_y, model, likelihood)

    # elif mode == 'Eval':
    #     print("start to test the model")
        predictions = eval_superpixels(model, likelihood)
        plot_result(predictions)
    else:
        raise Exception("No such mode")


if __name__== "__main__":
  main()