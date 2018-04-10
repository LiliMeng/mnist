# Basic MNIST Example

```bash

python gp_optimization.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```
## masked images for GP training. 

### Training data and label:

Currently we sample 1000 times for correct predictions as the training data. 

Training label is confidence score

Here 122 represents the index number from the 1000 imgs, [ 0.93749815] represents confidence score

masked_img_122_[ 0.93749815].png
