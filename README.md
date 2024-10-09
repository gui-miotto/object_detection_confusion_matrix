# Multiclass Confusion Matrix for Object Detection

An implementation using [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/).

## Introduction

Implementation of the multi-class confusion matrix for object detection as a [Metric](https://lightning.ai/docs/torchmetrics/stable/pages/quickstart.html) for [TorchMetrics](https://github.com/Lightning-AI/torchmetrics).

It builds on top of the [MulticlassConfusionMatrix](https://lightning.ai/docs/torchmetrics/stable/classification/confusion_matrix.html#multiclassconfusionmatrix) defined by TorchMetrics but which is restricted to classification problems.

Confusion matrices in object detection are a bit more tricky to calculate than in classification. That is because we have to deal with missed and spurious detections, or, in other words, false negative bounding boxes and false positive bounding boxes, respectively. This [article](https://medium.com/@tenyks_blogger/multiclass-confusion-matrix-for-object-detection-6fc4b0135de6) explains it quite well.

In this implementation, `ObjectDetectionConfusionMatrix` adds a class (i.e. something like a "Background" class) to the problem. This class is used to represent the false positives/negatives. It will occupy the last row and column of the confusion matrix.

## Usage

Here is a simple example:


```python
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
```

Which should print:

```bash
tensor([[1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]])
{'classes': tensor([0, 1, 2], dtype=torch.int32),
 'map': tensor(0.0757),
 'map_50': tensor(0.2525),
 'map_75': tensor(0.),
 'map_large': tensor(-1.),
 'map_medium': tensor(-1.),
 'map_per_class': tensor(-1.),
 'map_small': tensor(0.0757),
 'mar_1': tensor(0.0750),
 'mar_10': tensor(0.0750),
 'mar_100': tensor(0.0750),
 'mar_100_per_class': tensor(-1.),
 'mar_large': tensor(-1.),
 'mar_medium': tensor(-1.),
 'mar_small': tensor(0.0750)}
```

Notice that the confusion matrix is 4x4, not 3x3.