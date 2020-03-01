"""Module for (demo) viewer."""

import os
from dataclasses import dataclass
from glob import glob
from logging import getLogger
from os.path import basename, join
from typing import List, Optional, Tuple

import cv2
import numpy as np
import seaborn as sns
import torch
import torch.cuda
import torchvision
from hydra.utils import to_absolute_path

from frcnn.labels import COCO91
from frcnn.models import FasterRCNN, fasterrcnn_resnet50_fpn

__all__ = ["ImageViewer"]

logger = getLogger(__name__)

ColorType = Tuple[int, int, int]


@dataclass
class BasicConfig:
    gpu: bool
    conf: float
    display: bool
    weights: Optional[str]


@dataclass
class ImageConfig:
    root: str
    outputs: str


@dataclass
class Config:
    basic: BasicConfig
    image: ImageConfig


@dataclass
class FasterRCNNOutput:
    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor


class ImageViewer:
    COLORS: List[ColorType] = [
        tuple(int(c * 255) for c in color) for color in sns.color_palette(n_colors=len(COCO91))  # type: ignore
    ]

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._model = self._load_model(cfg.basic.weights)
        self._paths = sorted(glob(join(to_absolute_path(cfg.image.root), "*")))
        self._device = "cuda" if cfg.basic.gpu and torch.cuda.is_available() else "cpu"

        os.makedirs(cfg.image.outputs, exist_ok=True)

    @torch.no_grad()
    def run(self):
        self._model = self._model.to(self._device).eval()

        for i, path in enumerate(self._paths):
            image_bgr: np.ndarray = cv2.imread(path)
            image_rgb: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_tensor: torch.Tensor = torchvision.transforms.functional.to_tensor(image_rgb).to(self._device)

            # only the first element because input only one image
            output = FasterRCNNOutput(**self._model([image_tensor])[0])

            boxes = output.boxes.cpu().numpy()
            labels = output.labels.cpu().numpy()
            scores = output.scores.cpu().numpy()

            logger.debug(
                f"[{i + 1}/{len(self._paths)}] Detect {len([s for s in scores if s >= self._cfg.basic.conf]):2d} "
                + f"objects in {path}",
            )

            image_bgr = self._draw_results(image_bgr, boxes, labels, scores)

            if self._cfg.basic.display:
                cv2.imshow("", image_bgr)
                cv2.waitKey(1)

            cv2.imwrite(join(self._cfg.image.outputs, basename(path)), image_bgr)

    @staticmethod
    def _load_model(weights: Optional[str]) -> FasterRCNN:
        logger.debug(f"Load weights: {weights}")

        if weights is None:
            model = fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            model = fasterrcnn_resnet50_fpn(pretrained=False)
            model = model.load_state_dict(torch.load(weights))

        return model

    def _draw_results(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Draw texts and rectangles to the image (BGR)."""

        for box, label, score in zip(boxes, labels, scores):
            if score < self._cfg.basic.conf:
                continue

            image = cv2.putText(
                image,
                COCO91[label],
                (round(box[0]), round(box[1])),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=self.COLORS[label],
                thickness=2,
            )

            image = cv2.rectangle(
                image,
                (round(box[0]), round(box[1])),
                (round(box[2]), round(box[3])),
                color=self.COLORS[label],
                thickness=2,
            )

        return image
