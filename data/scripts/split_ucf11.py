#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

import argparse
import csv
import os
import sys

from catalyst import dl
import imageio
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from tqdm import tqdm


def load_groups(input_folder):
    """
    Load the list of sub-folders into a python list with their
    corresponding label.
    """
    groups = []
    label_folders = os.listdir(input_folder)
    index = 0
    for label_folder in sorted(label_folders):
        label_folder_path = os.path.join(input_folder, label_folder)
        if os.path.isdir(label_folder_path):
            group_folders = os.listdir(label_folder_path)
            for group_folder in group_folders:
                if group_folder != "Annotation":
                    groups.append([os.path.join(label_folder_path, group_folder), index])
            index += 1
    # macos hotfix
    groups = [[key, value] for key, value in groups if ".DS_Store" not in key]
    return groups


def img_to_tensor(im, normalize=None):
    tensor = torch.from_numpy(
        np.moveaxis(im / (255.0 if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32)
    )
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def process_group(input_folder, groups, indices, ranges, file_ext):
    engine = dl.DeviceEngine()
    resnet = torchvision.models.resnet18(pretrained=True)
    resnet.fc = nn.Sequential(nn.Flatten(), nn.Identity())
    resnet.train(False)
    resnet = engine.sync_device(resnet)

    videos = []
    features = []
    for i in tqdm(ranges):
        group = groups[indices[i]]
        video_files = os.listdir(group[0])
        for video_file in video_files:
            video_file_path = os.path.join(group[0], video_file)
            if os.path.isfile(video_file_path):
                video_file_path = os.path.abspath(video_file_path)
                ext = os.path.splitext(video_file_path)[1]
                if ext == file_ext:
                    # make sure we have enough frames and the file isn't corrupt
                    video_reader = imageio.get_reader(video_file_path, "ffmpeg")
                    if len(video_reader) >= 16:
                        images = [img_to_tensor(im).unsqueeze_(0) for im in video_reader]
                        if len(images) < 100 or len(images) > 300:
                            continue
                        video_file_path = video_file_path.replace(
                            os.path.abspath(input_folder) + "/", ""
                        )
                        videos.append([video_file_path, group[1], len(images)])
                        for batch in chunks(images, 32):
                            batch = torch.cat(batch)
                            batch = engine.sync_device(batch)
                            batch = resnet(batch).cpu().detach().numpy()
                            features.append(batch)
    features = np.vstack(features)
    return videos, features


def split_data(input_folder, groups, file_ext):
    """
    Split the data at random for train, eval and test set.
    """
    group_count = len(groups)
    indices = np.arange(group_count)

    np.random.seed(0)  # Make it deterministic.
    np.random.shuffle(indices)

    # 80% training and 20% test.
    train_count = int(0.8 * group_count)
    test_count = group_count - train_count

    train_csv, train_npy = process_group(
        input_folder, groups, indices, range(train_count), file_ext
    )
    valid_csv, valid_npy = process_group(
        input_folder, groups, indices, range(train_count, train_count + test_count), file_ext
    )

    return train_csv, train_npy, valid_csv, valid_npy


def write_to_csv(items, file_path):
    """
    Write file path and its target pair in a CSV file format.
    """
    if sys.version_info[0] < 3:
        with open(file_path, "wb") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            for item in items:
                writer.writerow(item)
    else:
        with open(file_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            for item in items:
                writer.writerow(item)


def main(input_folder, output_folder):
    """
    Main entry point, it iterates through all the video files in a folder or through all
    sub-folders into a list with their corresponding target label. It then split the data
    into training set and test set.
    :param input_folder: input folder contains all the video contents.
    :param output_folder: where to store the result.
    """
    groups = load_groups(input_folder)
    train_csv, train_npy, valid_csv, valid_npy = split_data(input_folder, groups, ".mpg")

    write_to_csv(train_csv, os.path.join(output_folder, "train.csv"))
    np.save(os.path.join(output_folder, "train.npy"), train_npy)
    write_to_csv(valid_csv, os.path.join(output_folder, "valid.csv"))
    np.save(os.path.join(output_folder, "valid.npy"), valid_npy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="Input folder containing the raw data.",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        help="Output folder for the generated training, validation and test text files.",
        required=True,
    )

    args = parser.parse_args()

    main(args.input_folder, args.output_folder)
