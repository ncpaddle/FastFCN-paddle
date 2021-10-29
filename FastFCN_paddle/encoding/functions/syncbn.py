##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Synchronized Cross-GPU Batch Normalization functions"""
from paddle.autograd import PyLayer

__all__ = ['normalization']


class Normalization(PyLayer):
    @staticmethod
    def forward(ctx, input, mean, inv_std, gamma, beta):
        ctx.save_for_backward(input, mean, inv_std, gamma, beta)

        return (input - mean.unsqueeze(-1)).mul_((inv_std*gamma).unsqueeze(-1)).add_(beta.unsqueeze(-1))

    @staticmethod
    def backward(ctx, gradOutput):
        input, mean, inv_std, gamma, beta = ctx.saved_variables

        gradInputMean = gradOutput * (inv_std*gamma).unsqueeze(-1)
        gradInput = gradInputMean
        gradMean = gradInputMean.sum((0, 2)).mul_(-1)

        gradInvStdGamma = (input - mean.unsqueeze(-1)).mul_(gradOutput).sum((0, 2))
        gradInvStd = gradInvStdGamma * gamma
        gradGamma = gradInvStdGamma * inv_std

        gradBeta = gradOutput.sum((0, 2))

        return gradInput, gradMean, gradInvStd, gradGamma, gradBeta


def normalization(input, mean, inv_std, gamma, beta):
    r"""Applies Batch Normalization over a 3d input that is seen as a
    mini-batch.

    .. _encoding.normalization:

    .. math::

        y = \frac{x - \mu[x]}{ \sqrt{var[x] + \epsilon}} * \gamma + \beta

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    """
    return Normalization.apply(input, mean, inv_std, gamma, beta)

if __name__ == '__main__':
    import paddle

    input = paddle.randn((3,4,5), dtype=paddle.float64, requires_grad=True).cuda()
    mean = paddle.randn((input.size(1),), dtype=paddle.float64, requires_grad=True).cuda()
    inv_std = paddle.randn((input.size(1),), dtype=paddle.float64, requires_grad=True).cuda()
    gamma = paddle.randn((input.size(1),), dtype=paddle.float64, requires_grad=True).cuda()
    beta = paddle.randn((input.size(1),), dtype=paddle.float64, requires_grad=True).cuda()

    assert paddle.autograd.gradcheck(normalization, (input, mean, inv_std, gamma, beta))
