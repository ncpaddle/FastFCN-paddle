##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Package Core NN Modules."""
import paddle
import paddle.nn.functional as F
import sys,os

from paddle.nn import Layer

# from ..functions import scaled_l2, aggregate
# from ..functions import scaled_l2, aggregate

__all__ = ['Encoding']

class Encoding(Layer):
    r"""
    Encoding Layer: a learnable residual encoder.

    .. image:: _static/img/cvpr17.svg
        :width: 50%
        :align: center

    Encoding Layer accpets 3D or 4D inputs.
    It considers an input featuremaps with the shape of :math:`C\times H\times W`
    as a set of C-dimentional input features :math:`X=\{x_1, ...x_N\}`, where N is total number
    of features given by :math:`H\times W`, which learns an inherent codebook
    :math:`D=\{d_1,...d_K\}` and a set of smoothing factor of visual centers
    :math:`S=\{s_1,...s_K\}`. Encoding Layer outputs the residuals with soft-assignment weights
    :math:`e_k=\sum_{i=1}^Ne_{ik}`, where

    .. math::

        e_{ik} = \frac{exp(-s_k\|r_{ik}\|^2)}{\sum_{j=1}^K exp(-s_j\|r_{ij}\|^2)} r_{ik}

    and the residuals are given by :math:`r_{ik} = x_i - d_k`. The output encoders are
    :math:`E=\{e_1,...e_K\}`.

    Args:
        D: dimention of the features or feature channels
        K: number of codeswords

    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}` or
          :math:`\mathcal{R}^{B\times D\times H\times W}` (where :math:`B` is batch,
          :math:`N` is total number of features or :math:`H\times W`.)
        - Output: :math:`E\in\mathcal{R}^{B\times K\times D}`

    Attributes:
        codewords (Tensor): the learnable codewords of shape (:math:`K\times D`)
        scale (Tensor): the learnable scale factor of visual centers

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

        Hang Zhang, Jia Xue, and Kristin Dana. "Deep TEN: Texture Encoding Network."
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017*

    Examples:
        >>> import encoding
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from torch.autograd import Variable
        >>> B,C,H,W,K = 2,3,4,5,6
        >>> X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), requires_grad=True)
        >>> layer = encoding.Encoding(C,K).double().cuda()
        >>> E = layer(X)
    """
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        # self.reset_params()
        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords = paddle.create_parameter(shape=[K, D], dtype='float32',
                                                 default_initializer=paddle.fluid.initializer.UniformInitializer(
                                                     low=-std1, high=std1))
        self.scale = paddle.create_parameter(shape=[K], dtype='float32',
                                             default_initializer=paddle.fluid.initializer.UniformInitializer(low=-1,
                                                                                                             high=0))

    # def reset_params(self):
    #     std1 = 1./((self.K*self.D)**(1/2))
    #     self.codewords.data.uniform_(-std1, std1)
    #     self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.shape[1] == self.D)
        B, D = X.shape[0], self.D
        if X.dim() == 3:
            # BxDxN => BxNxD
            X= paddle.transpose(X, perm=[0,2,1])
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW => Bx(HW)xD
            X = paddle.reshape(X, [B, D, -1])
            X = paddle.transpose(X, perm=[0, 2, 1])
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        A = F.softmax(self.scaled_l2(X, self.codewords, self.scale), axis=2)
        # aggregate
        E = self.aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, channels = codewords.shape
        batch_size = x.shape[0]
        reshaped_scale = paddle.reshape(scale, (1, 1, num_codes))
        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.shape[1], num_codes, channels))
        reshaped_codewords = paddle.reshape(codewords, (1, 1, num_codes, channels))

        scaled_l2_norm = reshaped_scale * (
            expanded_x - reshaped_codewords).pow(2).sum(axis=3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, channels = codewords.shape
        reshaped_codewords = paddle.reshape(codewords, (1, 1, num_codes, channels))
        batch_size = x.shape[0]
        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.shape[1], num_codes, channels))
        encoded_feat = (assignment_weights.unsqueeze(3) *
                        (expanded_x - reshaped_codewords)).sum(axis=1)
        return encoded_feat