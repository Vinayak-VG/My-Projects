# NeRF−−: Neural Radiance Fields Without Known Camera Parameters

### Pytorch Implementation of NeRF-- for 3D Reconstruction : [NeRF--](https://github.com/Vinayak-VG/My-Projects/tree/main/Computer_Vision_Projects/3D_Reconstruction/NeRF--)

## Summary

### Introduction

The original NeRF baseline uses camera parameters as input to the model but in a real situation we never have the access to the camera parameters and hence the original NeRF is not a suitable model to deploy in a real-world situation. 

This paper introduces NeRF-- which basically eliminates the need for camera parameters by approximating them during training. We see that the camera parameters including Intrinsic and Extrinsic parameters are estimated in a Joint optimisation problem and the model is able to predict the camera parameters even in the cases when COLMAP fails to estimate the camera parameters. 

The task is to reconstruct a 3D scene using 2D images and NeRF-- does this by aiming to approximate a dense light field from only sparse observations, such as a small set of images captured from diverse viewpoints.

### Previous Work

NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis - [Link](https://arxiv.org/abs/2003.08934v2)

### NeRF--

<img width="1231" alt="Screenshot 2021-11-18 at 10 43 29 PM" src="https://user-images.githubusercontent.com/80670240/142757660-9e1a8a09-ffc2-421d-9a45-4728ca4a222d.png">

<img width="919" alt="Screenshot 2021-11-21 at 2 58 53 PM" src="https://user-images.githubusercontent.com/80670240/142757672-32bab260-3ec8-432b-88bf-78450cc70e55.png">

Given a set of images I captured from 𝑁 sparse viewpoints of a scene, with their associated camera parameters Π, including both intrinsic and extrinsic, the goal of novel view synthesis is to come up with a scene representation that enables the generation of realistic images from the novel, unseen viewpoints. 

The NeRF model does this by representing the 3d volume space using a continuous function. Since it can’t predict accurately, it tries to approximate the continuous function 𝐹 : (𝒙, 𝒅) → (𝒄, 𝜎) with a Neural Network since it is a Universal Approximater. Here x represents the (x, y, z) of the point and d represents the viewing function (𝜃, 𝜙) which maps to radiance colour 𝒄 = (𝑟,𝑔,𝑏) and a density value 𝜎. 

To render an image from a NeRF model, the colour at each pixel 𝒑 = (𝑢,𝑣) on the image plane(𝐼^) is obtained by a rendering function R, aggregating the radiance along a ray shooting from the camera position 𝒐𝑖, passing through the pixel 𝒑 into the volume

𝐼^(𝒑) = R(𝒑,𝜋 |Θ) = ∫𝑇(h)𝜎 (𝒓(h))𝒄 (𝒓(h),𝒅) 𝑑h

𝑇 (h) = exp (- ∫𝜎(𝒓(𝑠))𝑑𝑠 )        

where the limits are from hn to hf where hn represents the near bound and hf represents the far bound

T(h) represents accumulated transmittance along the ray, i.e., the probability of the ray travelling from h𝑛 to h without hitting any other particle, and 𝒓 (h) = 𝒐 + h𝒅 denotes the camera ray that starts from camera origin 𝒐 and passes through 𝒑, controlled by the camera parameter Π, with near and far bounds hn and hf
Using the rendering function, we can now estimate the image from a novel view and calculate the photometric loss and also the loss for the camera parameters. 
Θ∗, Π∗ = argmin L(I^, Π^|I) where I^ and Π^ represent the predicted Image and camera parameters and Θ∗ and Π∗ represent the optimal parameters for the model, for the function and for the camera parameters respectively. 

### Camera Parameters

Camera Parameters are split into two parts 
- Camera Intrinsic - These depend on the type of camera we use. Ex: Focal Length
- Camera Extrinsic - These depend on space and determine position and orientation. Ex: Translation and Rotation 

#### Camera Intrinsic

<img width="191" alt="Screenshot 2021-11-18 at 11 22 19 PM" src="https://user-images.githubusercontent.com/80670240/142757680-ea38d66d-8fb7-43a6-b0c6-0e5332f86d4d.png">

𝑓𝑥 and 𝑓𝑦 denote camera focal lengths along the width and the height of the sensor respectively, and 𝑐𝑥 and 𝑐𝑦 denote principle points in the image plane.
We assume cx = W/2 and cy = H/2, and hence we need to only optimise fx and fy which we initialise them to be W and H respectively and train them

#### Camera Extrinsic

<img width="759" alt="Screenshot 2021-11-18 at 11 37 20 PM" src="https://user-images.githubusercontent.com/80670240/142757685-fbefae83-8c3d-408c-92de-8733d063953d.png">

We have translation t ∈ R3and hence it can be easily optimised while training but the rotation matrix R is a 3x3 matrix and hence we find a parameter 𝜙 which parameterizes as 𝜙 = 𝛼w where w is the normalised rotation axis and 𝛼 is the rotation angle

The set of camera parameters that we directly optimise in our model are the camera intrinsics 𝑓𝑥 and 𝑓𝑦 shared by all input images, and the camera extrinsic parameterised by 𝝓𝑖 and 𝒕𝑖 specific to each image 𝐼𝑖.

While Training for initialisation, the cameras for all input images are located at origin looking towards −𝑧-axis, i.e. all 𝑅𝑖 are initialised with identity matrices and all 𝑡𝑖 with zero vectors, and the focal lengths 𝑓𝑥 and 𝑓𝑦 are initialised to be the width 𝑊 and the height 𝐻 respectively, i.e. FOV≈ 53◦

**To Summarise**

<img width="766" alt="Screenshot 2021-11-18 at 11 41 01 PM" src="https://user-images.githubusercontent.com/80670240/142757721-07446dc3-edda-499d-a505-5e102aaa271f.png">

### Dataset 

We evaluate on [LLFF Dataset](https://www.robots.ox.ac.uk/~ryan/nerfmm2021/nerfmm_release_data.tar.gz), 8 classes and in each class around 12-40 images have been provided.

### Accuracy Metric 

For Novel-View Rendering
- Peak Signal-to-Noise Ratio (PSNR), 
- Structural Similarity Index Measure (SSIM)
- Learned Perceptual Image Patch Similarity (LPIPS)

For Camera Parameters
- As ground truth is not accessible for real scenes, we can only evaluate the accuracy by computing the difference between our optimised camera and the estimations obtained from COLMAP
- For focal length - Absolute Error 

For camera poses
- Absolute Trajectory Error (ATE) 

### Results
Author’s Result: 

<img width="1162" alt="Screenshot 2021-11-18 at 11 48 39 PM" src="https://user-images.githubusercontent.com/80670240/142757713-9971bbcd-ab90-4d60-be90-f3464e46410d.png">

### EndNote
To get to know more about the model, check out the paper: [NeRF--](https://arxiv.org/abs/2102.07064)

---

[Vinayak Gupta](https://github.com/Vinayak-VG)
21st Nov 2021




