from pprint import pprint

import torch

from object_detection_confusion_matrix import ObjectDetectionConfusionMatrix
from torchmetrics.detection import MeanAveragePrecision


def main():
    # Create some dummy data
    preds = [
        {
            "boxes": torch.tensor([[11, 11, 19, 19], [50, 50, 55, 55], [0, 0, 10, 10]]),
            "labels": torch.tensor([0, 2, 1]),
            "scores": torch.tensor([0.9, 0.8, 0.6])  # Not necessary but allows you to also calc mAP
        }
    ]
    target = [
        {
            "boxes": torch.tensor([[10, 10, 20, 20], [50, 50, 60, 60], [0, 0, 10, 10]]),
            "labels": torch.tensor([0, 2, 0])
        }
    ]
    # Calculate and print the confusion matrix.
    odcm = ObjectDetectionConfusionMatrix(num_classes=3)
    odcm.update(preds, target)
    pprint(odcm.compute())
    # Calculate and print the mean average precision metrics. Notice that
    # ObjectDetectionConfusionMatrix and MeanAveragePrecision have update() methods
    # with the same signature.
    map = MeanAveragePrecision()
    map.update(preds, target)
    pprint(map.compute())


if __name__ == "__main__":
    main()
