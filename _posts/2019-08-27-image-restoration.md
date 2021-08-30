---
layout: post
title: "Infinite Resolution Image processing with GAN and RL from unpaired image datasets."
author: "Abhishek"
categories: journal
tags: [documentation,sample]
image: mountains.jpg
---

# Contents

1. [Objective](#objective)
2. [Pre-processing Data](#pre-processing)
3. [Model](#model)
4. [Output](#output)
5. [Conclusion](#conclusion)

## Objective :
We present in this project a deep learning approach that is trained on unpaired data, namely a set of photographs that exhibits a retouching style the user likes, which is much easier to collect. Our system is formulated using deep convolutional neural networks that learn to apply different retouching operations on an input image. Network training with respect to various types of edits is enabled by modeling these retouching operations in a unified manner as resolution-independent differentiable filters. To apply the filters in a proper sequence and with suitable parameters, we employ a deep reinforcement learning approach that learns to make decisions on what action to take next, given the current state of the image. In contrast to many deep learning systems, ours provides users with an understandable solution in the form of conventional retouching edits, rather than just a "black-box" result. Through quantitative comparisons and user studies, we show that this technique generates retouching results consistent with the provided photo set.	

## Pre-processing

We used the [MIT-Adobe fiveK](https://data.csail.mit.edu/graphics/fivek/) dataset.

* 5,000 photos in DNG format
* This format can be read by DCRAW, Adobe Lightroom, and many other tools. An Adobe Lightroom catalog with renditions by 5 experts
* This includes values of individual sliders and the full history of adjustments for each photo.
* Semantic information about each photo Indoor vs. outdoor, time of day, type of light, and main subject (people, nature, man-made objects, etc)

Data augmentation and downsampling for image features is done as the original dataset is 50GB in size and requires Adobe lightroom to pre-process RAW files. The code can be found in [fetch_fiveK.py](https://github.com/Abhishek-Gawande/exposure/blob/master/fetch_fivek.py)

### Model
![alt text](/blog/assets/img/our-approach.png)

Our differentiable model consists of image features. We define a parent filter class and each of our features will inherit these parent class.

```python

class Filter:

  def __init__(self, net, cfg):
    self.cfg = cfg
    self.height, self.width, self.channels = list(map(int, net.get_shape()[1:]))

    # Specified in child classes
    self.num_filter_parameters = None
    self.short_name = None
    self.filter_parameters = None

  def get_short_name(self):
    assert self.short_name
    return self.short_name

  def get_num_filter_parameters(self):
    assert self.num_filter_parameters
    return self.num_filter_parameters

  def extract_parameters(self, features):
    output_dim = self.get_num_filter_parameters(
    ) + self.get_num_mask_parameters()
    features = ly.fully_connected(
        features,
        self.cfg.fc1_size,
        scope='fc1',
        activation_fn=lrelu,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    features = ly.fully_connected(
        features,
        output_dim,
        scope='fc2',
        activation_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    return features[:, :self.get_num_filter_parameters()], \
           features[:, self.get_num_filter_parameters():]

  # Should be implemented in child classes
  def filter_param_regressor(self, features):
    assert False

  # Process the whole image, without masking
  # Should be implemented in child classes
  def process(self, img, param):
    assert False

  def debug_info_batched(self):
    return False

  def no_high_res(self):
    return False

```

### GAN architecture

![alt text](/assets/img/gan_diagram.svg)

Both the generator and the discriminator are neural networks. The generator output is connected directly to the discriminator input. Through backpropagation, the discriminator's classification provides a signal that the generator uses to update its weights.

We used [WGAN](https://arxiv.org/abs/1701.07875v3) here.
Also used Monte-carlo based Reinforcement Learning approach.

the model picks a image operation, estimate the parameters from the Retouched images provided in the dataset.

![alt](/assets/img/image-op.png)

#### Output:
![alt text](/assets/img/steps.gif)

## Conclusion

This method provides following significant advantages over other peers:

* Resolution independant features
* Human understandable result
* Unpaired training

Limitations :
* RL/GAN stability 
* Hyperparameter Tuning
* poor performance on faces

This Was My Final Year Project Done with [Jayesh ](https://jayeshjawade.github.io)

The code can be found [here.](https://github.com/Abhishek-Gawande/exposure)
