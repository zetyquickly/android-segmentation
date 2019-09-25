from . import utils2
import numpy as np
import cv2
from caffe2.python import workspace
import os

PIXEL_MEANS_DEFAULT = np.array([[[0.0, 0.0, 0.0]]])
PIXEL_STDS_DEFAULT = np.array([[[1.0, 1.0, 1.0]]])

def preprocess(fname, blobs_dir_path, target_min_size, target_max_size):

    image = cv2.imread(fname)
    if image is None:
        return None
    
    target_size = target_max_size
    max_size = target_max_size
    rle_encode=True
    pixel_means=PIXEL_MEANS_DEFAULT
    pixel_stds=PIXEL_STDS_DEFAULT

    inputs = utils2.prepare_blobs(
        image,
        target_size=target_size,
        max_size=max_size,
        pixel_means=pixel_means,
        pixel_stds=pixel_stds,
    )
    for k, v in inputs.items():
        workspace.CreateBlob(k)
        workspace.FeedBlob(k, v)
        proto = workspace.SerializeBlob(k)
        with open(os.path.join(blobs_dir_path,'{}'.format(k)), 'wb') as f:
            f.write(proto)
