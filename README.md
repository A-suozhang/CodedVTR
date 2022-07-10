# CodedVTR: Codebook-based Sparse Voxel Transformer with Geometric Guidance(CVPR22)

Tianchen Zhao, Niansong Zhang, Xuefei Ning, He Wang, Li Yi\*, Yu Wang

The code implemention of [codedvtr](https://arxiv.org/abs/2203.09887), for motr information, please check our project page:
[https://a-suozhang.xyz/codedvtr.github.io/](https://a-suozhang.xyz/codedvtr.github.io/)


![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20220710144704.png)

# Installation

- please refer to the [setup.md](./setup.md)

# Example

to train the model on semantic-kitti

```
./kitti-train.sh $EXP_NAME $GPU_ID
```

for multi-gpu training

```
./kitti-train-mp.sh test 0,1
```

# Acknowledgment

This repository is developed based on [chrischoy/SpatioTemporalSegmentation-ScanNet](https://github.com/chrischoy/SpatioTemporalSegmentation-ScanNet)

# Citation 

if you find this project useful in your research, please consifer citing:

```
@inproceedings{zhao2022codedvtr,
  title={CodedVTR: Codebook-based Sparse Voxel Transformer with Geometric Guidance},
  author={Zhao, Tianchen and Zhang, Niansong and Ning, Xuefei and Wang, He and Yi, Li and Wang, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1435--1444},
  year={2022}
}
```

