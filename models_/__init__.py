# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import os
import sys
import pathlib
# test_path = os.environ["TEST_PATH"]

# if test_path is not None:

    # parent = pathlib.Path(test_path).parent.absolute()
    # child = pathlib.PurePath(test_path).name
    # sys.path.append(str(parent))

    # exec('from {}.models import resnet as resnet'.format(child))
    # exec('from {}.models import res16unet as res16unet'.format(child))
    # exec('from {}.models import mink_transformer as mink_transformer'.format(child))
    # exec('from {}.models import mink_transformer_voxel as mink_transformer_voxel'.format(child))
    # exec('from {}.models import point_transformer as point_transformer'.format(child))
    # exec('from {}.models import mixed_transformer as mixed_transformer'.format(child))

# else:
    # import models.resnet as resnet
    # import models.res16unet as res16unet
    # import models.mink_transformer as mink_transformer
    # import models.mink_transformer_voxel as mink_transformer_voxel
    # import models.point_transformer as point_transformer
    # import models.mixed_transformer as mixed_transformer

# import models.resnet as resnet
import models.res16unet as res16unet
# import models.mink_transformer as mink_transformer
# import models.mink_transformer_voxel as mink_transformer_voxel
# import models.point_transformer as point_transformer
# import models.mixed_transformer as mixed_transformer


MODELS = []


def add_models(module):
  MODELS.extend([getattr(module, a) for a in dir(module) if ('Net' in a or 'Transformer' in a)])


# add_models(resnet)
add_models(res16unet)
# add_models(mink_transformer)
# add_models(point_transformer)
# add_models(mink_transformer_voxel)
# add_models(mixed_transformer)

def get_models():
  '''Returns a tuple of sample models.'''
  return MODELS

def load_model(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  # Find the model class from its name
  all_models = get_models()
  mdict = {model.__name__: model for model in all_models}
  if name not in mdict:
    print('Invalid model index. Options are:')
    # Display a list of valid model names
    for model in all_models:
      print('\t* {}'.format(model.__name__))
    return None
  NetClass = mdict[name]

  return NetClass
