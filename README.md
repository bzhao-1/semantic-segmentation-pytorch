# Semantic Segmentation on CARLA Data
# Adapted from MIT ADE20K dataset in PyTorch

This is a PyTorch implementation of semantic segmentation models on MIT ADE20K scene parsing dataset (http://sceneparsing.csail.mit.edu/).


## Background of MIT ADE20K
ADE20K is the largest open source dataset for semantic segmentation and scene parsing, released by MIT Computer Vision team. Follow the link below to find the repository for their dataset and implementations on Caffe and Torch7:
https://github.com/CSAILVision/sceneparsing


### ADE20K Methods on Syncronized Batch Normalization on PyTorch
This module computes the mean and standard-deviation across all devices during training. We empirically find that a reasonable large batch size is important for segmentation. We thank [Jiayuan Mao](http://vccy.xyz/) for his kind contributions, please refer to [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) for details.



### State-of-the-Art models
- **PSPNet** is scene parsing network that aggregates global representation with Pyramid Pooling Module (PPM). It is the winner model of ILSVRC'16 MIT Scene Parsing Challenge. Please refer to [https://arxiv.org/abs/1612.01105](https://arxiv.org/abs/1612.01105) for details.
- **UPerNet** is a model based on Feature Pyramid Network (FPN) and Pyramid Pooling Module (PPM). It doesn't need dilated convolution, an operator that is time-and-memory consuming. *Without bells and whistles*, it is comparable or even better compared with PSPNet, while requiring much shorter training time and less GPU memory. Please refer to [https://arxiv.org/abs/1807.10221](https://arxiv.org/abs/1807.10221) for details.
- **HRNet** is a recently proposed model that retains high resolution representations throughout the model, without the traditional bottleneck design. It achieves the SOTA performance on a series of pixel labeling tasks. Please refer to [https://arxiv.org/abs/1904.04514](https://arxiv.org/abs/1904.04514) for details.


## Supported models
We split our models into encoder and decoder, where encoders are usually modified directly from classification networks, and decoders consist of final convolutions and upsampling. We have provided some pre-configured models in the ```config``` folder.

Encoder:
- MobileNetV2dilated
- ResNet18/ResNet18dilated
- ResNet50/ResNet50dilated
- ResNet101/ResNet101dilated
- HRNetV2 (W48)

Decoder:
- C1 (one convolution module)
- C1_deepsup (C1 + deep supervision trick)
- PPM (Pyramid Pooling Module, see [PSPNet](https://hszhao.github.io/projects/pspnet) paper for details.)
- PPM_deepsup (PPM + deep supervision trick)
- UPerNet (Pyramid Pooling + FPN head, see [UperNet](https://arxiv.org/abs/1807.10221) for details.)



## Reference

Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso and A. Torralba. International Journal on Computer Vision (IJCV), 2018. (https://arxiv.org/pdf/1608.05442.pdf)

    @article{zhou2018semantic,
      title={Semantic understanding of scenes through the ade20k dataset},
      author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Xiao, Tete and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
      journal={International Journal on Computer Vision},
      year={2018}
    }

Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. (http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)

    @inproceedings{zhou2017scene,
        title={Scene Parsing through ADE20K Dataset},
        author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2017}
    }
    
