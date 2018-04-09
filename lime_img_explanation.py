import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray, label2rgb # since the code wants color images

from sklearn.datasets import fetch_mldata
import cv2
from utils import save_checkpoint
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import cv2

from utils import weight_init
from utils import save_checkpoint
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer 
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from ImageExplanation import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

use_random_forest = False
use_neural_networks = True
train_nn = False

if use_random_forest == True:

	mnist = fetch_mldata('MNIST original')
	# make each image color so lim image works correctly
	X_vec = np.stack([gray2rgb(iimg) for iimg in mnist.data.reshape((-1, 28, 28))], 0)
	y_vec = mnist.target.astype(np.uint8)

	class PipeStep(object):
		"""
		Wrapper for turning functions into pipeline transforms (no-fitting)
		"""
		def __init__(self, step_func):
			self._step_func = step_func
		def fit(self, *args):
			return self
		def transform(self, X):
			return self._step_func(X)

	makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
	flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

	simple_rf_pipeline = Pipeline([
		('Make Gray', makegray_step),
		('Flatten Image', flatten_step),
		('RF', RandomForestClassifier())])


	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec,
	                                                    train_size=0.55)
	simple_rf_pipeline.fit(X_train, y_train)
	img = X_test[0]
	segmenter = slic(img, n_segments=50, compactness=10, sigma=1)

	explainer = LimeImageExplainer(verbose = False)

	explanation = explainer.explain_instance(X_vec[0], 
	                                         classifier_fn = simple_rf_pipeline.predict_proba, 
	                                         top_labels=10, hide_color=0, num_samples=10000, segmentation_fn=segmenter)


	fig, m_axs = plt.subplots(2,5, figsize = (12,6))

	for i, c_ax in enumerate(m_axs.flatten()):
	    temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=10, hide_rest=False, min_weight = 0.01 )
	    cv2.imshow('mask for {}'.format(i), mask*255*255)
	    cv2.imshow('orginal_img', X_test[0])
	    cv2.imshow('Positive for {}\nActual {}'.format(i, y_test[0]), label2rgb(mask,X_test[0], bg_label = 0))
	    cv2.waitKey(0)


elif use_neural_networks == True:
	
	kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
	train_loader = torch.utils.data.DataLoader(
	    datasets.MNIST('../data', train=True, download=True,
	                   transform=transforms.Compose([
	                       transforms.ToTensor(),
	                       #transforms.Normalize((0.1307,), (0.3081,))
	                   ])),
		batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
	    datasets.MNIST('../data', train=False, transform=transforms.Compose([
	                       transforms.ToTensor(),
	                       #transforms.Normalize((0.1307,), (0.3081,))
	                   ])),
	    batch_size=args.test_batch_size, shuffle=True, **kwargs)


	def conv( inp_chl, out_chl, ker_size = 3, stride = 1, padding = 1 ):
	    return nn.Sequential(
	        nn.Conv2d( inp_chl, out_chl, ker_size, stride = stride, padding = padding ),
	        nn.BatchNorm2d( out_chl ),
	        nn.ReLU( True ),
	        )

	def tconv( inp_chl, out_chl, ker_size = 4, stride = 2, padding = 1 ):
	    return nn.Sequential(
	        nn.ConvTranspose2d( inp_chl, out_chl, ker_size, stride = stride, padding = padding ),
	        nn.BatchNorm2d( out_chl ),
	        nn.ReLU( True ),
	        )

	class Classification_Net( nn.Module ):
	    def __init__(self):
	        super().__init__()
	        self.conv1 = conv( 1, 32 )
	        self.conv2 = conv( 32, 32 )
	        self.conv3 = conv( 32, 64, stride = 2 )
	        self.conv4 = conv( 64, 64 )
	        self.conv5 = conv( 64, 128, stride = 2 )
	        self.conv6 = nn.Conv2d( 128, 128, 3, padding = 1 )
	        self.fc1 = nn.Linear( 128, 10 )

	    def forward( self, x ):
	        x0 = self.conv2( self.conv1( x  ) )
	        x1 = self.conv4( self.conv3( x0 ) )
	        x2 = self.conv6( self.conv5( x1 ) )

	        f = x2.mean(3).mean(2)
	        pred0 = self.fc1( f )
	       
	        return x0, x1, x2, pred0

	model = Classification_Net()
	if args.cuda:
	    model.cuda()

	optimizer_cls = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)
	
	def train_cls(epoch):
	    model.train()
	    for batch_idx, (data, target) in enumerate(train_loader):
	        if args.cuda:
	            data, target = data.cuda(), target.cuda()
	        data, target = Variable(data), Variable(target)
	        optimizer_cls.zero_grad()
	        x0, x1, x2, pred0 = model(data)
	        output = F.log_softmax(pred0, dim=1)

	        loss = F.nll_loss(output, target)
	        loss.backward()
	        optimizer_cls.step()
	        if batch_idx % args.log_interval == 0:
	            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
	                epoch, batch_idx * len(data), len(train_loader.dataset),
	                100. * batch_idx / len(train_loader), loss.data[0]))

	def eval_cls():
	    model.eval()
	    test_loss = 0
	    correct = 0
	    for data, target in test_loader:
	        if args.cuda:
	            data, target = data.cuda(), target.cuda()
	        data, target = Variable(data, volatile=True), Variable(target)
	        x0, x1, x2, pred0 = model(data)
	        output = F.log_softmax(pred0, dim=1)
	        test_loss += F.nll_loss(output, target, size_average=False).data[0]
	        pred = output.data.max(1, keepdim=True)[1]
	        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
	       
	    test_loss /= len(test_loader.dataset)
	    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	        test_loss, correct, len(test_loader.dataset),
	        100. * correct / len(test_loader.dataset)))

	if train_nn == True:
		for epoch in range(1, 5):
			train_cls(epoch)
			eval_cls()
			save_checkpoint({
	                    'epoch': epoch,
	                    'model': model.state_dict(),
	                    #'optimizer_cls': optimizer_cls.state_dict(),
	                    #'optimizer_reg': optimizer_reg.state_dict(),
	                }, is_best=False, save_folder="saved_checkpoints" , filename='checkpoint.pth.tar')
	else:
		eval_cls()
else:
	raise Exception("not implemented yet")

