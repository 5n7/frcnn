"""Module for Faster R-CNN models.
Currently uses the implementation by torchvision official.
By setting pretrained to True, the weights pretrained with COCO of 91 classes are automatically downloaded.
"""

from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn

__all__ = ["FasterRCNN", "fasterrcnn_resnet50_fpn"]
