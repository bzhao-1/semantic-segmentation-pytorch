# Semantic Segmentation on CARLA Data for CV4AD Carleton College Senior Thesis 
# Adapted from MIT ADE20K dataset in PyTorch

This is a PyTorch implementation of semantic segmentation models on custom data from the CARLA simulator across 4 different weather scenarios. Each dataset contains 1995 total images with an 80-10-10 split for training, testing, and validation. 
The implementation is adapted from the MIT ADE20K scene parsing dataset (http://sceneparsing.csail.mit.edu/).


## Background of MIT ADE20K
ADE20K is the largest open source dataset for semantic segmentation and scene parsing, released by MIT Computer Vision team. Follow the link below to find the repository for their dataset and implementations on Caffe and Torch7:
https://github.com/CSAILVision/sceneparsing

### State-of-the-Art models
- **HRNet** is a recently proposed model that retains high resolution representations throughout the model, without the traditional bottleneck design. It achieves the SOTA performance on a series of pixel labeling tasks. Please refer to [https://arxiv.org/abs/1904.04514](https://arxiv.org/abs/1904.04514) for details.


## Models

We trained the model on CARLA data with a HRNETV2 encoder. This was adpated from the configurations provided in the ADE20K source code. 

Encoder:
- HRNetV2


Decoder:
- C1 (one convolution module)




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
    
