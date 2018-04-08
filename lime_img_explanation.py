import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray, label2rgb # since the code wants color images

from sklearn.datasets import fetch_mldata
import cv2
from nn_train_eval import *
mnist = fetch_mldata('MNIST original')

# make each image color so lim image works correctly
X_vec = np.stack([gray2rgb(iimg) for iimg in mnist.data.reshape((-1, 28, 28))], 0)
y_vec = mnist.target.astype(np.uint8)



from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer 
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from ImageExplanation import *

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

if random_forest == True:

	X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec,
	                                                    train_size=0.55)
	simple_rf_pipeline.fit(X_train, y_train)
	img = X_test[0]
elif neural_networks == True:
	for epoch in range(1, 11):
		train_cls(epoch)
   	 	eval_cls()
        save_checkpoint({
                    'epoch': epoch,
                    'model': model.state_dict(),
                }, is_best=False, save_folder="saved_checkpoints" , filename='checkpoint.pth.tar')
    

else:
	raise Exception("not implemented yet")

segmenter = slic(img, n_segments=50, compactness=10, sigma=1)

#cv2.imshow('segmented_img', mark_boundaries(img, segmenter))
#cv2.waitKey(0)

explainer = LimeImageExplainer(verbose = False)
explanation = explainer.explain_instance(X_vec[0], 
                                         classifier_fn = simple_rf_pipeline.predict_proba, 
                                         top_labels=10, hide_color=0, num_samples=10000, segmentation_fn=segmenter)

fig, m_axs = plt.subplots(2,5, figsize = (12,6))

for i, c_ax in enumerate(m_axs.flatten()):
    temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=10, hide_rest=False, min_weight = 0.01 )
    cv2.imshow('mask for {}'.format(i), mask*255*255)
    cv2.imshow('orginal_img', X_test[0])
    #cv2.imshow('Positive for {}\nActual {}'.format(i, y_test[0]), label2rgb(mask,X_test[0], bg_label = 0))
    cv2.waitKey(0)
