import numpy as np

def intersect_and_union(pred_label, label, num_classes, ignore_index):
    """Calculate intersection and Union.
    Args:
        pred_label (ndarray): Prediction segmentation map
        label (ndarray): Ground truth segmentation map
        num_classes (int): Number of categories.

     Returns:
         ndarray: The intersection of prediction and ground truth
            histogram on all classes.
         ndarray: The union of prediction and ground truth histogram on
            all classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = np.load(label)

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]
        
    intersect = pred_label[pred_label == label]
    area_intersect = np.histogram(intersect, bins=np.arange(num_classes + 1))[0]
    area_pred_label = np.histogram(pred_label, bins=np.arange(num_classes + 1))[0]
    area_label = np.histogram(label, bins=np.arange(num_classes + 1))[0]

    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results, gt_seg_maps, num_classes, ignore_index):
    """Calculate Total Intersection and Union.
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros(num_classes, dtype=np.int32)
    total_area_union = np.zeros(num_classes, dtype=np.int32)
    total_area_pred_label = np.zeros(num_classes, dtype=np.int32)
    total_area_label = np.zeros(num_classes, dtype=np.int32)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(results[i], gt_seg_maps[i],
                                                                                      num_classes, ignore_index)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, total_area_label


def mean_iou(results, gt_seg_maps, num_classes, ignore_index):
    """Calculate Mean Intersection and Union (mIoU)
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category IoU, shape (num_classes, ).
    """
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps, num_classes, ignore_index)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    iou = total_area_intersect / total_area_union
    return all_acc, acc, iou