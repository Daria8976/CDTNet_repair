# CDTNet-High-Resolution-Image-Harmonization


This is the official repository for the following paper:

> **High-Resolution Image Harmonization via Collaborative Dual Transformations**  [[arXiv]](https://arxiv.org/abs/2109.06671)<br>
>
> Wenyan Cong, Xinhao Tao, Li Niu, Jing Liang, Xuesong Gao, Qihao Sun, Liqing Zhang<br>
> Accepted by **CVPR2022**.

We propose a high-resolution image harmonization network named **CDTNet** to combine pixel-to-pixel transformation and RGB-to-RGB transformation coherently in an end-to-end framework. As shown in the image below, our CDTNet consists of a low-resolution generator for pixel-to-pixel transformation, a color mapping module for RGB-to-RGB transformation, and a refinement module to take advantage of both. 

<div align="center">
  <img src='https://bcmi.sjtu.edu.cn/~niuli/github_images/CDTnetwork.jpg' align="center" width=800>
</div>

<br>Unfortunately, code and model are not allowed to be released due to the collaboration with Hisense, but our network could be easily reimplemented based on the public [code sources](#codesource). We provide two datasets used in our paper. We will provide the harmonization results of our method for comparison. More results for comparison are available upon request. 

## Datasets

### 1. HAdobe5k

HAdobe5k is one of the four synthesized sub-datasets in [iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4) dataset, which is the benchmark dataset for image harmonization. Specifically, HAdobe5k is generated based on [MIT-Adobe FiveK](<http://data.csail.mit.edu/graphics/fivek/>) dataset and contains 21597 image triplets (composite image, real image, mask) as shown below, where 19437 triplets are used for training and 2160 triplets are used for test. Official training/test split could be found in **[Baidu Cloud](https://pan.baidu.com/s/1NAtLnCdY1-4uxRKB8REPQg)** [**(Alternative_address)**](https://cloud.bcmi.sjtu.edu.cn/sharing/eBXLV8iU5).

MIT-Adobe FiveK provides with 6 retouched versions for each image, so we manually segment the foreground region and exchange foregrounds between 2 versions to generate composite images. High-resolution images in HAdobe5k sub-dataset are with random resolution from 1296 to 6048, which could be downloaded from **[Baidu Cloud](https://pan.baidu.com/s/1NAtLnCdY1-4uxRKB8REPQg)** [**(Alternative_address)**](https://cloud.bcmi.sjtu.edu.cn/sharing/eBXLV8iU5).

<img src='examples/hadobe5k.png' align="center" width=600>

### 2. 100 High-Resolution Real Composite Images

Considering that the composite images in HAdobe5k are synthetic composite images, we additionally provide 100 high-resolution real composite images for qualitative comparison in real scenarios with image pairs (composite image, mask), which are generated based on [Open Image Dataset V6](https://storage.googleapis.com/openimages/web/index.html) dataset and [Flickr](https://www.flickr.com). 

Open Image Dataset V6 contains ~9M images with 28M instance segmentation annotations of 350 categories, where enormous images are collected from Flickr and with high resolution. So the foreground images are collected from the whole Open Image Dataset V6, where the provided instance segmentations are used to crop the foregrounds. The background images are collected from both Open Image Dataset V6 and Flickr, considering the resolutions and semantics. Then cropped foregrounds and background images are combined using PhotoShop, leading to obviously inharmonious composite images.

100 high-resolution real composite images are with random resolution from 1024 to 6016, which could be downloaded from **[Baidu Cloud](https://pan.baidu.com/s/1fTfLBMxb7sAKtbpQVsfh8g)** (access code: vnrp) [**(Alternative_address)**](https://cloud.bcmi.sjtu.edu.cn/sharing/c9frU77Il).

<img src='examples/hr_real_comp_100.jpg' align="center" width=700>

## Acknowledgement<a name="codesource"></a> 

Our code is heavily borrowed from [iSSAM](https://github.com/saic-vul/image_harmonization) and [3D LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).
