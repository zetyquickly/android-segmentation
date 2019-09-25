from . import utils2
from caffe2.python import workspace
import os
import numpy as np
import cv2


def postprocess(image_fname, blobs_dir_path):

    image = cv2.imread(image_fname)
    if image is None:
        return None  

    out_data = []
    for item in ["class_nms", "score_nms", "bbox_nms", "mask_fcn_probs"]:
        with open(os.path.join(blobs_dir_path, "{}".format(item)), 'rb') as f:
            proto = f.read()
        blob = workspace.DeserializeBlob(proto)
        out_data.append(blob.fetch())
    

    classids = out_data[0]
    scores = out_data[1]  # bbox scores, (R, )
    boxes = out_data[2]  # i.e., boxes, (R, 4*1)
    masks = out_data[3]  # (R, cls, mask_dim, mask_dim)

    rle_encode = True
    R = boxes.shape[0]
    im_masks = []
    if R > 0:
        im_dims = image.shape
        im_masks = utils2.compute_segm_results(
            masks, boxes, classids, im_dims[0], im_dims[1], rle_encode=rle_encode
        )

    boxes = np.column_stack((boxes, scores))

    ret = {"classids": classids, "boxes": boxes, "masks": masks, "im_masks": im_masks}
    return ret