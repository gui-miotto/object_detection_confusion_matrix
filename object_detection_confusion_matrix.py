from typing import List, Dict

import torch
from torchmetrics.classification import MulticlassConfusionMatrix

from torchvision.ops import box_iou


class ObjectDetectionConfusionMatrix(MulticlassConfusionMatrix):
    """
    Calculates the Multiclass Confusion Matrix for object detection.
    In object detection, differently than classification, the number of predictions doesn't
    necessarily match the number of targets. The missmatch in these numbers are caused by
    missed detections and spurious detections, or, in other words, false negative bounding boxes
    and false positive bounding boxes, respectively.
    ObjectDetectionConfusionMatrix inserts an extra class, the background class, to the list of
    classes in the object detection problem. The background class is used to represent the misses
    and spurious detections. Numbers for the background class are shown in the last row and
    column of the confusion matrix.
    """

    def __init__(self, iou_threshold: float = 0.5, **kwargs):
        self.iou_thresh = iou_threshold
        self.background_id = kwargs.pop("num_classes")
        # Add one to the number of class to accomodade the background class
        super().__init__(num_classes=self.background_id + 1, **kwargs)

    def update(self, preds: List[Dict[str, torch.Tensor]], target: List[Dict[str, torch.Tensor]]):
        if len(preds) != len(target):
            raise ValueError("len(preds) != len(target)")
        # Update the confusion matrix for each image individually
        for img_pred, img_target in zip(preds, target):
            self._update_one_image(preds=img_pred, target=img_target)

    def _update_one_image(self, preds: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        dev = target["labels"].device
        # Calculate IoU for all possible combinations of predicted and target bounding boxes
        iou_matrix = box_iou(boxes1=preds["boxes"], boxes2=target["boxes"])
        target_match_iou, target_match_id = iou_matrix.max(dim=0)
        iou_mask = target_match_iou >= self.iou_thresh
        # Now we start paring the the predicted and target class labels.
        # These pairings can be of three types:
        # 1 - True positive, i.e. when pred and target bboxes have large enough overlap
        # 2 - False negatives, i.e. when a target bbox has no large enough overlap with a pred bbox
        # 3 - False positives, i.e. when a pred bbox has no large enough overlap with a target bbox
        # Note: "large enough overlap" means IoU > iou_thresh
        expd_pred_labels, expd_target_labels = [], []
        # Legit matches:
        matched_preds = target_match_id[iou_mask]
        expd_pred_labels.append(preds["labels"][matched_preds])
        expd_target_labels.append(target["labels"][iou_mask])
        # False negatives:
        false_negatives = ~iou_mask
        n_fnegatives = false_negatives.sum().item()
        if n_fnegatives:
            expd_pred_labels.append(torch.tensor(n_fnegatives * [self.background_id], device=dev))
            expd_target_labels.append(target["labels"][false_negatives])
        # False positives:
        false_positive_ids = set(range(len(preds["labels"]))) - set(matched_preds.cpu().numpy())
        n_fpositives = len(false_positive_ids)
        if n_fpositives:
            expd_pred_labels.append(preds["labels"][torch.tensor(tuple(false_positive_ids))])
            expd_target_labels.append(torch.tensor(n_fpositives * [self.background_id], device=dev))
        # Put everything together
        expd_pred_labels = torch.concat(expd_pred_labels)
        expd_target_labels = torch.concat(expd_target_labels)
        # Call parent's class update()
        super().update(preds=expd_pred_labels, target=expd_target_labels)
