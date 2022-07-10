# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.

from models.model import Model, NetworkType
from models.modules.common import ConvType, NormType, conv, conv_tr, get_norm, get_nonlinearity_fn
from models.modules.resnet_block import BasicBlock, Bottleneck, SingleConv, SingleChannelConv
from models.modules.tr_block import *

import torch
from torch import nn
import numpy as np
import MinkowskiEngine.MinkowskiOps as me

class mySequential(nn.Sequential):
    def forward(self, inputs, *args):
        for module in self._modules.values():
            if len(args) == 1:
                pass
            # if type(inputs) == tuple:
                # inputs = module(*inputs)
            # else:
            inputs = module(inputs, *args)
        return inputs


class ResNetBase(Model):
  BLOCK = None
  LAYERS = ()
  INIT_DIM = 64
  PLANES = (64, 128, 256, 512)
  OUT_PIXEL_DIST = 32
  HAS_LAST_BLOCK = False
  CONV_TYPE = ConvType.HYPERCUBE

  def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
    assert self.BLOCK is not None
    assert self.OUT_PIXEL_DIST > 0

    super(ResNetBase, self).__init__(in_channels, out_channels, config, D, **kwargs)

    self.network_initialization(in_channels, out_channels, config, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, out_channels, config, D):

    def space_n_time_m(n, m):
      return n if D == 3 else [n, n, n, m]

    if D == 4:
      self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

    dilations = config.dilations
    bn_momentum = config.bn_momentum
    self.inplanes = self.INIT_DIM
    self.conv1 = conv(
        in_channels,
        self.inplanes,
        kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
        stride=1,
        D=D)

    self.bn1 = get_norm(NormType.BATCH_NORM, self.inplanes, D=self.D, bn_momentum=bn_momentum)
    self.relu = ME.MinkowskiReLU(inplace=True)
    self.pool = sum_pool(kernel_size=space_n_time_m(2, 1), stride=space_n_time_m(2, 1), D=D)

    self.layer1 = self._make_layer(
        self.BLOCK,
        self.PLANES[0],
        self.LAYERS[0],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[0], 1))
    self.layer2 = self._make_layer(
        self.BLOCK,
        self.PLANES[1],
        self.LAYERS[1],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[1], 1))
    self.layer3 = self._make_layer(
        self.BLOCK,
        self.PLANES[2],
        self.LAYERS[2],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[2], 1))
    self.layer4 = self._make_layer(
        self.BLOCK,
        self.PLANES[3],
        self.LAYERS[3],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[3], 1))

    if self.NETWORK_TYPE == NetworkType.CLASSIFICATION:
      self.glob_avg = ME.MinkowskiGlobalPooling(dimension=D)
      if self.HAS_LAST_BLOCK:
        self.final1 = nn.Linear(self.inplanes, self.inplanes, bias=False)
        self.bnfinal1 = nn.BatchNorm1d(self.inplanes)

        self.final2 = nn.Linear(self.inplanes, self.inplanes, bias=False)
        self.bnfinal2 = nn.BatchNorm1d(self.inplanes)

      self.final = nn.Linear(self.inplanes, out_channels, bias=True)
    else:
      self.final = conv(
          self.PLANES[3] * self.BLOCK.expansion, out_channels, kernel_size=1, bias=True, D=D)

  def weight_initialization(self):
    for m in self.modules():
      if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _make_layer(self,
                  block,
                  planes,
                  blocks,
                  stride=1,
                  dilation=1,
                  norm_type=NormType.BATCH_NORM,
                  nonlinearity_type='ReLU',
                  bn_momentum=0.1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv(
              self.inplanes,
              planes * block.expansion,
              kernel_size=1,
              stride=stride,
              bias=False,
              D=self.D),
          get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum),
      )
    layers = []
    layers.append(
        block(
            self.inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            conv_type=self.CONV_TYPE,
            nonlinearity_type=nonlinearity_type,
            D=self.D))
    self.inplanes = planes * block.expansion 
    for i in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              stride=1,
              dilation=dilation,
              conv_type=self.CONV_TYPE,
              nonlinearity_type=nonlinearity_type,
              D=self.D))

    return mySequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.pool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    if self.NETWORK_TYPE == NetworkType.CLASSIFICATION:
      if self.HAS_LAST_BLOCK:
        res = self.glob_avg(x)
        x = self.final1(res)
        x = self.bnfinal1(x)
        x = self.relu(x)

        x = self.final2(x)
        x = self.bnfinal2(x)
        x += res
        x = self.relu(x)
      else:
        x = self.glob_avg(x)

    x = self.final(x)
    return x



class Res16UNetBase(ResNetBase):
  BLOCK = None
  PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
  DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
  INIT_DIM = 32
  OUT_PIXEL_DIST = 1
  NORM_TYPE = NormType.BATCH_NORM
  NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
  # CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS
  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
    super(Res16UNetBase, self).__init__(in_channels, out_channels, config, D)

  def network_initialization(self, in_channels, out_channels, config, D):

    if not isinstance(self.BLOCK, list): # if single type
        self.BLOCK = [self.BLOCK]*len(self.PLANES)
    # print('IN CHANNEL inside the model {}'.format(in_channels))

    # Setup net_metadata
    dilations = self.DILATIONS
    bn_momentum = config.bn_momentum

    block_noexpansion = True

    def space_n_time_m(n, m):
      return n if D == 3 else [n, n, n, m]

    if D == 4:
      self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

    if config.xyz_input is not None:
        if config.xyz_input:
            in_channels = in_channels + 3

    if config.dataset == 'SemanticKITTI':
        in_channels = 4
    if config.dataset == 'S3DIS':
        in_channels = 9

    # Output of the first conv concated to conv6
    self.inplanes = self.INIT_DIM
    self.conv0p1s1 = conv(
        in_channels,
        self.inplanes,
        kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
        stride=1,
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)

    self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

    self.conv1p1s2 = conv(
        self.inplanes,
        self.inplanes if not block_noexpansion else self.PLANES[0],
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.inplanes = self.inplanes if not block_noexpansion else self.PLANES[0]
    self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

    self.block1 = self._make_layer(
        self.BLOCK[0],
        self.PLANES[0],
        self.LAYERS[0],
        dilation=dilations[0],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)

    self.conv2p2s2 = conv(
        self.inplanes,
        self.inplanes if not block_noexpansion else self.PLANES[1],
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.inplanes = self.inplanes if not block_noexpansion else self.PLANES[1]
    self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block2 = self._make_layer(
        self.BLOCK[1],
        self.PLANES[1],
        self.LAYERS[1],
        dilation=dilations[1],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)

    self.conv3p4s2 = conv(
        self.inplanes,
        self.inplanes if not block_noexpansion else self.PLANES[2],
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.inplanes = self.inplanes if not block_noexpansion else self.PLANES[2]
    self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block3 = self._make_layer(
        self.BLOCK[2],
        self.PLANES[2],
        self.LAYERS[2],
        dilation=dilations[2],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)

    self.conv4p8s2 = conv(
        self.inplanes,
        self.inplanes if not block_noexpansion else self.PLANES[3],
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.inplanes = self.inplanes if not block_noexpansion else self.PLANES[3]
    self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block4 = self._make_layer(
        self.BLOCK[3],
        self.PLANES[3],
        self.LAYERS[3],
        dilation=dilations[3],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)
    self.convtr4p16s2 = conv_tr(
        self.inplanes,
        self.PLANES[4],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr4 = get_norm(self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK[4].expansion
    self.block5 = self._make_layer(
        self.BLOCK[4],
        self.PLANES[4],
        self.LAYERS[4],
        dilation=dilations[4],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)
    self.convtr5p8s2 = conv_tr(
        self.inplanes,
        self.PLANES[5],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr5 = get_norm(self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK[5].expansion
    self.block6 = self._make_layer(
        self.BLOCK[5],
        self.PLANES[5],
        self.LAYERS[5],
        dilation=dilations[5],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)
    self.convtr6p4s2 = conv_tr(
        self.inplanes,
        self.PLANES[6],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr6 = get_norm(self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK[6].expansion
    self.block7 = self._make_layer(
        self.BLOCK[6],
        self.PLANES[6],
        self.LAYERS[6],
        dilation=dilations[6],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)
    self.convtr7p2s2 = conv_tr(
        self.inplanes,
        self.PLANES[7],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr7 = get_norm(self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[7] + self.INIT_DIM
    self.block8 = self._make_layer(
        self.BLOCK[7],
        self.PLANES[7],
        self.LAYERS[7],
        dilation=dilations[7],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)

    self.final = conv(self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True, D=D)

    if config.enable_point_branch:
      self.point_transform_mlp = nn.ModuleList([
        nn.Sequential(
          nn.Linear(self.INIT_DIM, self.PLANES[3]),
          # nn.BatchNorm1d(self.PLANES[3]),
          nn.ReLU(True)
        ),
        nn.Sequential(
          nn.Linear(self.PLANES[3], self.PLANES[5]),
          #nn.BatchNorm1d(self.PLANES[5]),
          nn.ReLU(True)
        ),
        nn.Sequential(
          nn.Linear(self.PLANES[5], self.PLANES[7]),
          # nn.BatchNorm1d(self.PLANES[7]),
          nn.ReLU(True)
        )
      ])
      self.downsample16x = nn.Sequential(
        ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3),
        ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3),
        ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3),
        ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
      )
      self.downsample4x = nn.Sequential(
        ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3),
        ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
      )
      self.dropout = nn.Dropout(0.1, True)
      # self.dropout = nn.Dropout(0.3, True)
      self.interpolate = ME.MinkowskiInterpolation(return_kernel_map=False, return_weights=False)
    

  def forward(self, x, save_anchor=False, iter_=None, aux=None, enable_point_branch=False):
    # for n, m in self.named_modules():
        # if 'block' in n:
            # if hasattr(m, "schedule_update"):
                # m.schedule_update(iter_)
    if save_anchor:
        self.anchors = []
    # mapped to transformer.stem1
    out = self.conv0p1s1(x)
    out = self.bn0(out)
    out_p1 = get_nonlinearity_fn(self.config.nonlinearity, out)

    if enable_point_branch:
      out_p1_point = out_p1.F
      out_p1_coord = out_p1.C

    # mapped to transformer.stem2
    out = self.conv1p1s2(out_p1)
    out = self.bn1(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)

    # mapped to transformer.PTBlock1
    out_b1p2 = self.block1(out, iter_, aux)
    if save_anchor:
        self.anchors.append(out_b1p2)

    # mapped to transformer.PTBlock2
    out = self.conv2p2s2(out_b1p2)
    out = self.bn2(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)
    out_b2p4 = self.block2(out, iter_, aux)
    if save_anchor:
        self.anchors.append(out_b2p4)

    # mapped to transformer.PTBlock3
    out = self.conv3p4s2(out_b2p4)
    out = self.bn3(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)
    out_b3p8 = self.block3(out, iter_, aux)
    # if save_anchor:
        # self.anchors.append(out_b3p8)

    # pixel_dist=16
    # mapped to transformer.PTBlock4
    out = self.conv4p8s2(out_b3p8)
    out = self.bn4(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)
    out = self.block4(out, iter_, aux)
    # if save_anchor:
        # self.anchors.append(out)

    if enable_point_branch:
      interpolated_out = self.interpolate(out, out_p1_coord.type(torch.FloatTensor).to(out.device))
      # fused feature
      block4_features = interpolated_out + self.point_transform_mlp[0](out_p1_point)
      out_fused = ME.SparseTensor(features=block4_features, coordinates=out_p1_coord)
      out_fused = self.downsample16x(out_fused)
      out = ME.SparseTensor(features=self.dropout(out_fused.F), coordinate_map_key=out.coordinate_map_key, coordinate_manager=out.coordinate_manager)

    # pixel_dist=8
    # mapped to transfrormer.PTBlock5
    out = self.convtr4p16s2(out)
    out = self.bntr4(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)
    out = me.cat(out, out_b3p8)
    out = self.block5(out, iter_, aux)
    # out = self.block5(out)
    # if save_anchor:
      # self.anchors.append(out)

    # pixel_dist=4
    # mapped to transformer.PTBlock6
    out = self.convtr5p8s2(out)
    out = self.bntr5(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)
    out = me.cat(out, out_b2p4)
    out = self.block6(out, iter_, aux)
    if save_anchor:
      self.anchors.append(out)

    if enable_point_branch:
      interpolated_out = self.interpolate(out, out_p1_coord.type(torch.FloatTensor).to(out.device))
      block6_features = interpolated_out + self.point_transform_mlp[1](block4_features)
      out_fused = ME.SparseTensor(features=block6_features, coordinates=out_p1_coord)
      out_fused = self.downsample4x(out_fused)
      out = ME.SparseTensor(features=self.dropout(out_fused.F), coordinate_map_key=out.coordinate_map_key, coordinate_manager=out.coordinate_manager)

    # pixel_dist=2
    # mapped to transformer.PTBlock7
    out = self.convtr6p4s2(out)
    out = self.bntr6(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)
    out = me.cat(out, out_b1p2)
    out = self.block7(out, iter_, aux)
    if save_anchor:
      self.anchors.append(out)

    # pixel_dist=1
    # mapped to transformer.final_conv
    out = self.convtr7p2s2(out)
    out = self.bntr7(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)

    out = me.cat(out, out_p1)
    out = self.block8(out, iter_, aux)

    if enable_point_branch:
      interpolated_out = self.interpolate(out, out_p1_coord.type(torch.FloatTensor).to(out.device))
      block8_features = interpolated_out + self.point_transform_mlp[2](block6_features)
      out_fused = ME.SparseTensor(features=block8_features, coordinates=out_p1_coord)
      out = ME.SparseTensor(features=self.dropout(out_fused.F), coordinate_map_key=out.coordinate_map_key, coordinate_manager=out.coordinate_manager)


    out = self.final(out)

    if torch.isnan(out.F).sum() > 0:
        import ipdb; ipdb.set_trace()

    if save_anchor:
        return out, self.anchors
    else:
        return out


class Res16UNet14(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class Res16UNet18(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class Res16UNet34(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

class Res16UNet50(Res16UNetBase):
  BLOCK = Bottleneck
  LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet101(Res16UNetBase):
  BLOCK = Bottleneck
  LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class Res16UNet14A(Res16UNet14):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet14A2(Res16UNet14A):
  LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B(Res16UNet14):
  PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet14B2(Res16UNet14B):
  LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B3(Res16UNet14B):
  LAYERS = (2, 2, 2, 2, 1, 1, 1, 1)


class Res16UNet14C(Res16UNet14):
  PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class Res16UNet14D(Res16UNet14):
  PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18A(Res16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet18B(Res16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet18D(Res16UNet18):
  PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18E(Res16UNet18):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet18F(Res16UNet18):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34A(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet34B(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34C(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

class Res16UNetTest(Res16UNetBase):
  BLOCK = [SingleConv]*8
  BLOCK[0] = SingleConv
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)

class Res16UNetTestA(Res16UNetTest):

  DEPTH_RATIO = 2
  PLANES_RATIO = 0.75

  BLOCK= [CodedVTRBlock]*8
  BLOCK[0]= BasicBlock
  BLOCK[-1]= BasicBlock

  LAYERS = (np.array([1, 1, 1, 1, 1, 1, 1, 1])*DEPTH_RATIO).astype(int)
  PLANES = (np.array([32, 64, 128, 256, 128, 128, 96, 96])*PLANES_RATIO).astype(int)

class Res18UNet(Res16UNetTest):
  BLOCK= [SingleConv]*8

  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
  PLANES = (np.array([32, 64, 128, 256, 256, 128, 96, 96])*1.0).astype(int)


class Res16UNet(Res16UNetBase):
  BLOCK= [SingleConv]*8

  LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
  PLANES = (np.array([32, 64, 128, 256, 256, 128, 96, 96])).astype(int)

class Res34UNet(Res16UNetBase):
  BLOCK= [SingleConv]*8
  LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)











