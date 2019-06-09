# Attention Branch Network for CelebA
Writer : [Hiroshi Fukui](https://github.com/Hiroshi-Fukui)

## Abstract
This repository contains the source code of Attention Branch Network for image classification. Detail of ABN is as follows:
[CVPR paper, ](http://openaccess.thecvf.com/content_CVPR_2019/html/Fukui_Attention_Branch_Network_Learning_of_Attention_Mechanism_for_Visual_Explanation_CVPR_2019_paper.html)
[ArXiv paper, ](https://arxiv.org/abs/1812.10025)
and, 
[ABN for image classification](https://github.com/machine-perception-robotics-group/attention_branch_network)


## Citation

```
@article{fukui2018,
	author = {Hiroshi Fukui and Tsubasa Hirakawa and Takayoshi Yamashita and Hironobu Fujiyoshi},
	title = {Attention Branch Network: Learning of Attention Mechanism for Visual Explanation},
	journal = {Computer Vision and Pattern Recognition},
	year = {2019},
	pages = {10705-10714}
}
```
```
@article{fukui2018,  
	author = {Hiroshi Fukui and Tsubasa Hirakawa and Takayoshi Yamashita and Hironobu Fujiyoshi},  
	title = {Attention Branch Network: Learning of Attention Mechanism for Visual Explanation},  
	journal = {arXiv preprint arXiv:1812.10025},  
	year = {2018}  
}  
```

## Detail
Our ABN for for CelebA is written by [chainer](https://github.com/chainer/chainer). 
Requirements of chainer version is as follows, and we published the [DockerHub](https://cloud.docker.com/u/fhiro0125/repository/docker/fhiro0125/chainercv_07_1).
- chainer : 2.1.0
- cupy : 1.0.3

run commands of train and test are follows:

> python train.py CelebA_Multi-task_Recognition_attention-sig_itr3/train_param.py

> python test.py CelebA_Multi-task_Recognition_attention-sig_itr3/train_param.py

