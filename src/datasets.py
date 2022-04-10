import os

import albumentations as albu
import cv2
import imageio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def to_torch_img(im):
    # todo: rewrite
    im = albu.smallest_max_size(im, 224, cv2.INTER_LINEAR)
    im = albu.center_crop(im, 224, 224)
    im = np.moveaxis(im / (255.0 if im.dtype == np.uint8 else 1), -1, 0).astype(
        np.float32
    )
    return im


# class GroupTransform:
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, images):
#         for tr in self.transforms:
#             p = getattr(tr, "p", 1.0)
#             if random.random() < p:
#                 params = getattr(tr, "get_params", lambda: {})()
#                 images = [tr.apply(x, **params) for x in images]
#         return images


class TemporalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_path: str,
        num_segments: int = None,
        segment_len: int = None,
    ):
        super().__init__()
        self.data = df.to_dict()
        self.root_path = root_path
        self.num_segments = num_segments
        self.segment_len = segment_len

    def __getitem__(self, idx):
        target = self.data["class"][idx]
        path = self.data["path"][idx]
        video_file_path = os.path.join(self.root_path, path)
        video_reader = imageio.get_reader(video_file_path, "ffmpeg")
        # todo: rewrite
        images = [im for im in video_reader]

        if self.num_segments is not None:
            images = np.array_split(images, self.num_segments)

        if self.segment_len is not None:
            sampled_images = []
            for images_segment in images:
                idxs = np.random.choice(
                    len(images_segment),
                    self.segment_len,
                    replace=len(images_segment) < self.segment_len,
                )
                sampled_images.extend(images_segment[idxs])
        else:
            sampled_images = images

        features = np.vstack([to_torch_img(im)[np.newaxis] for im in sampled_images])
        return features, target

    def __len__(self):
        return len(self.data["class"])

    def get_labels(self):
        return list(self.data["class"].values())
