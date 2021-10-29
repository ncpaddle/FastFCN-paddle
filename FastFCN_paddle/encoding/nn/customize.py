##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Custermized NN Module"""
import paddle
import paddle.nn as nn

from paddle.nn import functional as F
from paddle.nn import Layer, Sequential, Conv2D, ReLU, AdaptiveAvgPool2D, BCELoss

# from paddle.autograd import Variable

paddle_ver = paddle.__version__[:3]

__all__ = ['SegmentationLosses', 'PyramidPooling', 'JPU', 'JPU_X', 'Mean']


class CrossEntropyLoss(nn.Layer):
    """
    Implements the cross entropy loss function.

    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for
            the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining. Default ``1.0``.
        data_format (str, optional): The tensor format to use, 'NCHW' or 'NHWC'. Default ``'NCHW'``.
    """

    def __init__(self,
                 weight=None,
                 ignore_index=255,
                 top_k_percent_pixels=1.0,
                 data_format='NCHW'):
        super(CrossEntropyLoss, self).__init__()
        if weight is not None:
            weight = paddle.to_tensor(weight, dtype='float32')
        self.weight = weight
        self.ignore_index = ignore_index
        self.top_k_percent_pixels = top_k_percent_pixels
        self.EPS = 1e-8
        self.data_format = data_format

    def forward(self, logit, label, semantic_weights=None):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            semantic_weights (Tensor, optional): Weights about loss for each pixels, shape is the same as label. Default: None.
        """


        channel_axis = 1 if self.data_format == 'NCHW' else -1
        if self.weight is not None and logit.shape[channel_axis] != len(
                self.weight):
            raise ValueError(
                'The number of weights = {} must be the same as the number of classes = {}.'
                .format(len(self.weight), logit.shape[1]))

        if channel_axis == 1:
            logit = paddle.transpose(logit, [0, 2, 3, 1])
        label = label.astype('int64')

        loss = F.cross_entropy(
            logit,
            label,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.weight)

        mask = label != self.ignore_index
        mask = paddle.cast(mask, 'float32')

        loss = loss * mask
        if semantic_weights is not None:
            loss = loss * semantic_weights

        if self.weight is not None:
            _one_hot = F.one_hot(label, logit.shape[-1])
            coef = paddle.sum(_one_hot * self.weight, axis=-1)
        else:
            coef = paddle.ones_like(label)

        label.stop_gradient = True
        mask.stop_gradient = True
        if self.top_k_percent_pixels == 1.0:
            avg_loss = paddle.mean(loss) / (paddle.mean(mask * coef) + self.EPS)
            return avg_loss

        loss = loss.reshape((-1, ))
        top_k_pixels = int(self.top_k_percent_pixels * loss.numel())
        loss, indices = paddle.topk(loss, top_k_pixels)
        coef = coef.reshape((-1, ))
        coef = paddle.gather(coef, indices)

        coef.stop_gradient = True
        coef = coef.astype('float32')

        return loss.mean() / (paddle.mean(coef) + self.EPS)

class SegmentationLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""

    def __init__(self, se_loss=True, se_weight=0.2, nclass=19,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1, reduction='mean'):
        super(SegmentationLosses, self).__init__(weight=weight, ignore_index=ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight, reduction='mean')
    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as('float32')
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(nn.Sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass)

            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(paddle.fluid.layers.sigmoid(se_pred), se_target.astype('float32'))
            # print("loss1:", loss1)
            # print("loss2:", loss2)
            # print("loss3:", loss3)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3


    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.shape[0]
        tvect = paddle.zeros([batch, nclass], dtype='bool')
        for i in range(batch):
            hist = paddle.histogram(target[i],
                                    bins=nclass, min=0,
                                    max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect

        return tvect

class Normalize(Layer):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)


class PyramidPooling(Layer):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2D(1)
        self.pool2 = AdaptiveAvgPool2D(2)
        self.pool3 = AdaptiveAvgPool2D(3)
        self.pool4 = AdaptiveAvgPool2D(6)

        out_channels = int(in_channels/4)
        self.conv1 = Sequential(Conv2D(in_channels, out_channels, 1, bias_attr=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2D(in_channels, out_channels, 1, bias_attr=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2D(in_channels, out_channels, 1, bias_attr=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2D(in_channels, out_channels, 1, bias_attr=False),
                                norm_layer(out_channels),
                                ReLU(True))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return paddle.concat((x, feat1, feat2, feat3, feat4), 1)


class SeparableConv2D(nn.Layer):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2D):
        super(SeparableConv2D, self).__init__()

        self.conv1 = nn.Conv2D(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias_attr=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2D(inplanes, planes, 1, 1, 0, 1, 1, bias_attr=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Layer):
    def __init__(self, in_channels, width=512, norm_layer=None, up_kwargs=None):
        super(JPU, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2D(in_channels[-1], width, 3, padding=1, bias_attr=False),
            norm_layer(width),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2D(in_channels[-2], width, 3, padding=1, bias_attr=False),
            norm_layer(width),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2D(in_channels[-3], width, 3, padding=1, bias_attr=False),
            norm_layer(width),
            nn.ReLU())

        self.dilation1 = nn.Sequential(SeparableConv2D(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU())
        self.dilation2 = nn.Sequential(SeparableConv2D(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU())
        self.dilation3 = nn.Sequential(SeparableConv2D(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU())
        self.dilation4 = nn.Sequential(SeparableConv2D(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU())

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].shape
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feat = paddle.concat(feats, axis=1)
        feat = paddle.concat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], axis=1)

        return inputs[0], inputs[1], inputs[2], feat


class JUM(nn.Layer):
    def __init__(self, in_channels, width, dilation, norm_layer, up_kwargs):
        super(JUM, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv_l = nn.Sequential(
            nn.Conv2D(in_channels[-1], width, 3, padding=1, bias_attr=False),
            norm_layer(width),
            nn.ReLU())
        self.conv_h = nn.Sequential(
            nn.Conv2D(in_channels[-2], width, 3, padding=1, bias_attr=False),
            norm_layer(width),
            nn.ReLU())

        norm_layer = lambda n_channels: nn.GroupNorm(32, n_channels)
        self.dilation1 = nn.Sequential(SeparableConv2D(2*width, width, kernel_size=3, padding=dilation, dilation=dilation, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU())
        self.dilation2 = nn.Sequential(SeparableConv2D(2*width, width, kernel_size=3, padding=2*dilation, dilation=2*dilation, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU())
        self.dilation3 = nn.Sequential(SeparableConv2D(2*width, width, kernel_size=3, padding=4*dilation, dilation=4*dilation, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU())

    def forward(self, x_l, x_h):
        feats = [self.conv_l(x_l), self.conv_h(x_h)]
        _, _, h, w = feats[-1].shape
        feats[-2] = F.upsample(feats[-2], (h, w), **self.up_kwargs)
        feat = paddle.concat(feats, axis=1)
        feat = paddle.concat([feats[-2], self.dilation1(feat), self.dilation2(feat), self.dilation3(feat)], axis=1)

        return feat

class JPU_X(nn.Layer):
    def __init__(self, in_channels, width=512, norm_layer=None, up_kwargs=None):
        super(JPU_X, self).__init__()
        self.jum_1 = JUM(in_channels[:2], width//2, 1, norm_layer, up_kwargs)
        self.jum_2 = JUM(in_channels[1:], width, 2, norm_layer, up_kwargs)

    def forward(self, *inputs):
        feat = self.jum_1(inputs[2], inputs[1])
        feat = self.jum_2(inputs[3], feat)

        return inputs[0], inputs[1], inputs[2], feat


class Mean(Layer):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)
