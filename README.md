# RandLA-Net in robot system
This repository contains the implementation of [RandLA-Net (CVPR 2020 Oral)](https://arxiv.org/abs/1911.11236) in PyTorch.
- We only support SemanticKITTI dataset now. (Welcome everyone to develop together and raise PR)
- Our model is almost as good as the original implementation. (Validation set : Our 52.9% mIoU vs original 53.1%)
- We place our pretrain-model in [`pretrain_model/checkpoint.tar`](pretrain_model/checkpoint.tar) directory.



## Acknowledgement

- Original Tensorflow implementation [link](https://github.com/QingyongHu/RandLA-Net)
- Our network & config codes are modified from [RandLA-Net PyTorch](https://github.com/qiqihaer/RandLA-Net-pytorch)
- Our evaluation & visualization codes are modified from [SemanticKITTI API](https://github.com/PRBonn/semantic-kitti-api)
