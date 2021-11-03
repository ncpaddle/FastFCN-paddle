# FastFCN-paddle
Paddle implementation of FastFCN : A Faster, Stronger and Lighter framework for semantic segmentation

* Original Paper: [FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation](https://arxiv.org/pdf/1903.11816v1.pdf)

by Huikai Wu, Junge Zhang, Kaiqi Huang, Kongming Liang, Yizhou Yu.

* Original Pytorch Implementation: [https://github.com/wuhuikai/FastFCN](https://github.com/wuhuikai/FastFCN)

## Introduction  

We implement the Paper reproduction with [Paddle](https://github.com/PaddlePaddle/Paddle)  . The results  are close to the orginal paper on the ade20k datasets. 

A brief introduction about important folders:

`diff`: log and fake data from recommended procedures of reproducing research papers [论文复现指南](https://github.com/PaddlePaddle/models/blob/develop/docs/ThesisReproduction_CV.md#4)

`FastFCN-paddle`: paddle version of FastFCN

`FastFCN-python`: pytorch version of FastFCN

## Training:
```python
%cd FastFCN_paddle 
!python -m experiments.segmentation.train --dataset ade20k \ 
            --model encnet --jpu JPU --aux --se-loss \ 
            --backbone resnet50 --checkname encnet_res50_ade20k_train 
```
## Evaluating:
```python 
%cd FastFCN_paddle 
!python -m experiments.segmentation.test --dataset ade20k \
    --model encnet --jpu JPU --aux --se-loss \
    --backbone resnet50 --resume {MODEL} --split val --mode testval
```

## Results:  

| Method  |DataSet| Environment | Model| Epoch| Test Accuracy|   
| --- | --- |--- | --- |---|---|  
| FastFCN|ade20k| Tesla V100 | EncNet+JPU|  |  |  

## Reprod_Log:  

`forward_diff`: [forward_diff.log](https://github.com/ncpaddle/FastFCN-paddle/blob/master/diff/forward_diff.log)  
`metric_diff` : [metric_diff.log](https://github.com/ncpaddle/FastFCN-paddle/blob/master/diff/metric_diff.log)  
`loss_diff` : [loss_diff.log](https://github.com/ncpaddle/FastFCN-paddle/blob/master/diff/loss_diff.log)  
`bp_align_diff` : [bp_align_diff.log](https://github.com/ncpaddle/FastFCN-paddle/blob/master/diff/bp_align_diff.log)  
`train_align_diff` : [train_align_diff.log](https://github.com/ncpaddle/FastFCN-paddle/blob/master/diff/train_align_diff.log)  
`train_log` : [train.log](https://github.com/ncpaddle/FastFCN-paddle/blob/master/diff/train.log)
## AI studio:
* AI studio link : []() 
