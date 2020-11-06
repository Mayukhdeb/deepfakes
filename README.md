# deepfakes
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mayukhdeb/deepfakes/blob/master/notebooks/train_model_on_colab.ipynb)


<img src = "https://github.com/Mayukhdeb/deepfakes/blob/master/output/preds.jpg?raw=true">


I will arrange all of the notes properly once I get a better idea of how it works. 

## Important links
1. [Deepfake paper](https://arxiv.org/abs/1909.11573)
2. [Pixel shuffling paper](https://arxiv.org/pdf/1609.05158v2.pdf)
3. [Article on super-resolution](https://towardsdatascience.com/an-evolution-in-single-image-super-resolution-using-deep-learning-66f0adfb2d6b)
4. [Umeyama algorithm paper](https://pdfs.semanticscholar.org/d107/231cce2676dbeea87e00bb0c587c280b9c53.pdf) 
5. [Article on transformations of images](https://towardsdatascience.com/transformations-with-opencv-ff9a7bea7f8b)
2. [Real-Time Single Image and Video Super-Resolution Using an Efficient
Sub-Pixel Convolutional Neural Network i.e pixel shuffling paper](https://arxiv.org/pdf/1609.05158v2.pdf)
3. [Useful article in super-resolution](https://towardsdatascience.com/an-evolution-in-single-image-super-resolution-using-deep-learning-66f0adfb2d6b)
4. [Useful article to understand the concept of transformation matrices](https://towardsdatascience.com/transformations-with-opencv-ff9a7bea7f8b)

---
## Notes from the deepfake paper 

### The original method

This required three main parts:
1. An encoder that encodes an image into a lower dimensional vector.
2. A decoder that reproduces face A from the encoded vector
3. Another decoder that reproduces face B from the encoded vector. 

This image from the original paper summarises it better than anything else:

<img src = "images/paper_E_C.jpg" width = "50%">

---
## Notes from the pixel shuffling paper

### Intro
Generally, super-resolution operation is done on the high resolution space, but in this paper, they proposed a new way to do it in the low resolution space, which in turn would require less computational power. 

### More about SR operations
The SR operation is effectively a **one-to-many mapping from LR to HR space** which can have multiple solutions. A key assumption that underlies many SR techniques is that much of the high-frequency data is redundant and thus can be accurately reconstructed from low frequency components.

> Important full form: **PSNR** = **P**eak **S**ignal to **N**oise **R**atio. Higher means good quality and low means bad quality w.r.t the original image. It is measured in decibels (dB). Here's the [wikipedia page](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)

### Drawbacks of older approaches
* **Increasing the resolution** of the LR images before the image
enhancement step **increases the computational complexity**.
* **Interpolation methods** such as bicubic interpolation **do not bring additional information** to solve the problem. 


### What's new in their approach ?
Contrary to previous works, they increase the resolution from LR to HR only at the very end of the network and super-resolve HR data from LR feature maps. It's advantages are;
1. Requires lower computational power.
2. Not using an explicit interpolation filter means that the network implicitly learns the processing necessary for SR.

### Transpose convolutions v/s sub-pixel convolutions

In transpose convolutions, upsampling with strides adss zero values to upscale the image which are to be filled later on. Maybe even worse, these zero values have no gradient information that can be backpropagated through.

While sub-pixel convolutional layers essentially uses regular convolutional layers followed by a specific type of image reshaping called a **phase shift**. Instead of putting zeros in between pixels and having to do extra computation, they calculate more convolutions in lower resolution and resize the resulting map into an upscaled image. This way, no meaningless zeros are necessary. 

<img src = "https://raw.githubusercontent.com/atriumlts/subpixel/master/images/spcnn_diagram.png" width = "70%">

> Some parts written above are quoted from [this repository](https://github.com/atriumlts/subpixel)

**Phase shift** is also called “pixel shuffle”, which is a tensor of size `H × W × C · r² ` to size `rH × rW × C` tensor as shown below. An implementation of this operation can be found [here](https://github.com/Mayukhdeb/deepfakes/blob/8df7f7ace220b76f4f9564ddf444593d36a85f4f/deepfake_utils/autoencoder_utils.py#L52). 

<img src = "images/pixel_shuffle.png" width = "50%">

--- 
## Understanding the significance of the umeyama algorithm
### Mr. Shinji Umeyama asked:
**If 2 point patterns are given, what is the set of similarity transformation parameters that give the least mean squared error between the patterns ?**

And this is exactly what the umeyama algorithm does, it finds a set of similarity transformation parameters (rotation, translation, scaling) that minimizes the MSE Loss between the patterns.

<img src = "https://github.com/Mayukhdeb/deepfakes/blob/master/images/umeyama_on_image.png?raw=true" width = "60%">

**Note**
* The transformed pattern has the **minimum possible MSE loss** w.r.t the target pattern.
* The transformed pattern is **similar** to the source pattern. 

### How does it help here in deepfakes ? 

<img src = "https://github.com/Mayukhdeb/deepfakes/blob/master/images/download.png?raw=true">
We're generating a target image such that it's MSE loss w.r.t the distorted input image is minimized.Thanks to the umeyama algorithm, we are able to do this without distorting the key visual features.

**The two important points to note are:**
* The set of similarity transformations applied to the original image is such that the MSE loss between the target image and the distorted image has been minimized
* The target image is **similar** to the original image (i.e no distortions)


**Now what is a similarity transformation matrix ?**

It represents a set of operations that can be done on a matrix `A` to get another matrix `B` that is similar to `A`
Each value in a transformation matrix in 2D represents the following: 
```
[[size, rotation, location], ←x-axis
[rotation, size, location]] ←y-axis
```

A default matrix or one that wouldn’t change anything would look like;

```
[[1, 0, 0]
 [0, 1, 0]]
```

Or if we want to alter just the width (half the width along the x axis), it  would look like:

```
[[0.5, 0, 0], #x 
 [0, 1, 0]]   #y
```
    
We're taking only the first 2 indices of the transformation matrix because the third one represents the set of transformations required on the z axis, which would be `[0,0,1]`

## to-do:
1. ~~See what happens by feeding facial landmarks into the encoder-decoder model~~ Implemented without umeyama, need to implement again with umeyama
2. ~~Implement umeyama after resizing to `(256,256)`~~ Works 
3. ~~Figure out why umeyama works~~ Tried my best
4. ~~Integrate albumentations~~ Improved performance 
5. Reduce cropping of faces in generate_training_data.py, face_alignment is not able to detect the fake face landmarks. 
