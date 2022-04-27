import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import torch

from src.settings import DATA_ROOT


class TSQuantileTransformer:
    def __init__(self, *args, n_quantiles: int, **kwargs):
        self.n_quantiles = n_quantiles
        self._args = args
        self._kwargs = kwargs
        self.transforms = {}

    def fit(self, features: np.ndarray):
        for i in range(features.shape[1]):
            self.transforms[i] = QuantileTransformer(
                *self._args, n_quantiles=self.n_quantiles, **self._kwargs
            ).fit(features[:, i, :])
        return self

    def transform(self, features: np.ndarray):
        result = np.empty_like(features, dtype=np.int32)
        for i in range(features.shape[1]):
            result[:, i, :] = (
                self.transforms[i].transform(features[:, i, :]) * self.n_quantiles
            ).astype(np.int32)
        return result


# There are 2 different datasets COBRE and ABIDE,
# each contain two classes of subjects: patients and controls.
# There are multiple files to be read,
# and although they are all available in the repository,
# I am attaching what you need to reproduce a dataset for ABIDE.
# It will get you data in the shape: (569, 53, 140), where you have 569 subjects.
# Each has 53 channels with 140 time points.
# Learn from the code how to apply the same to COBRE.
def load_ABIDE1(
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("abide/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv"),
):
    hf = h5py.File(dataset_path, "r")
    data = hf.get("ABIDE1_dataset")
    data = np.array(data)
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)

    # take only those brain networks that are not noise
    df = pd.read_csv(indices_path, header=None)
    c_indices = df.values
    c_indices = c_indices.astype("int")
    c_indices = c_indices.flatten()
    c_indices = c_indices - 1
    finalData = data[:, c_indices, :]

    df = pd.read_csv(labels_path, header=None)
    labels = df.values.flatten() - 1

    return finalData, labels


def load_FBIRN(
    dataset_path: str = DATA_ROOT.joinpath("fbirn/FBIRN_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("fbirn/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("fbirn/labels_FBIRN_new.csv"),
):
    hf = h5py.File(dataset_path, "r")
    data = hf.get("FBIRN_dataset")
    data = np.array(data)
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)

    # take only those brain networks that are not noise
    df = pd.read_csv(indices_path, header=None)
    c_indices = df.values
    c_indices = c_indices.astype("int")
    c_indices = c_indices.flatten()
    c_indices = c_indices - 1
    finalData = data[:, c_indices, :]

    df = pd.read_csv(labels_path, header=None)
    labels = df.values.flatten() - 1

    return finalData, labels


def _find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


# taken from https://github.com/UsmanMahmood27/MILC
def load_ABIDE1_origin(
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5"),
    hc_path: str = DATA_ROOT.joinpath("abide/ABIDE1_HC_TrainingIndex.h5"),
    sz_path: str = DATA_ROOT.joinpath("abide/ABIDE1_SZ_TrainingIndex.h5"),
    indices_path: str = DATA_ROOT.joinpath("abide/correct_indices_GSP.csv"),
    index_array_path: str = DATA_ROOT.joinpath("abide/index_array_labelled_ABIDE1.csv"),
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv"),
):
    ID = 5
    ntrials = 1
    tr_sub = [15, 25, 50, 75, 100, 150]
    sub_per_class = tr_sub[ID]
    sample_x = 100
    sample_y = 20
    subjects = 569
    tc = 140
    samples_per_subject = 13
    n_val_HC = 50
    n_val_SZ = 50
    n_test_HC = 50
    n_test_SZ = 50
    window_shift = 10

    hf = h5py.File(dataset_path, "r")  # "../Data/ABIDE1_AllData.h5"
    data = hf.get("ABIDE1_dataset")
    data = np.array(data)
    data = data.reshape(subjects, sample_x, tc)

    # Get Training indices for cobre and convert them to tensor.
    # this is to have same training samples everytime.
    hf_hc = h5py.File(hc_path, "r")
    HC_TrainingIndex = hf_hc.get("HC_TrainingIndex")
    HC_TrainingIndex = np.array(HC_TrainingIndex)
    HC_TrainingIndex = torch.from_numpy(HC_TrainingIndex)

    hf_sz = h5py.File(sz_path, "r")
    SZ_TrainingIndex = hf_sz.get("SZ_TrainingIndex")
    SZ_TrainingIndex = np.array(SZ_TrainingIndex)
    SZ_TrainingIndex = torch.from_numpy(SZ_TrainingIndex)

    # if args.fMRI_twoD:
    #     finalData = data
    #     finalData = torch.from_numpy(finalData).float()
    #     finalData = finalData.permute(0, 2, 1)
    #     finalData = finalData.reshape(
    #         finalData.shape[0], finalData.shape[1], finalData.shape[2], 1
    #     )
    # else:
    finalData = np.zeros((subjects, samples_per_subject, sample_x, sample_y))
    for i in range(subjects):
        for j in range(samples_per_subject):
            finalData[i, j, :, :] = data[i, :, (j * window_shift) : (j * window_shift) + sample_y]
    finalData = torch.from_numpy(finalData).float()

    # print(finalData.shape)
    filename = indices_path
    # print(filename)
    df = pd.read_csv(filename, header=None)
    c_indices = df.values
    c_indices = torch.from_numpy(c_indices).int()
    c_indices = c_indices.view(53)
    c_indices = c_indices - 1
    finalData2 = finalData[:, :, c_indices.long(), :]

    filename = index_array_path
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).long()
    index_array = index_array.view(subjects)

    filename = labels_path
    df = pd.read_csv(filename, header=None)
    df = pd.read_csv(filename, header=None)
    all_labels = df.values
    all_labels = torch.from_numpy(all_labels).int()
    all_labels = all_labels.view(subjects)
    all_labels = all_labels - 1
    finalData2 = finalData2[index_array, :, :, :]
    all_labels = all_labels[index_array]

    HC_index, SZ_index = _find_indices_of_each_class(all_labels)

    total_HC_index_tr = HC_index[: len(HC_index) - (n_val_HC + n_test_HC)]
    total_SZ_index_tr = SZ_index[: len(SZ_index) - (n_val_SZ + n_test_SZ)]

    HC_index_val = HC_index[len(HC_index) - (n_val_HC + n_test_HC) : len(HC_index) - n_test_HC]
    SZ_index_val = SZ_index[len(HC_index) - (n_val_SZ + n_test_SZ) : len(HC_index) - n_test_SZ]

    HC_index_test = HC_index[len(HC_index) - (n_test_HC) :]
    SZ_index_test = SZ_index[len(SZ_index) - (n_test_SZ) :]

    # for trial in range(ntrials):
    trial = 0
    HC_random = HC_TrainingIndex[trial]
    SZ_random = SZ_TrainingIndex[trial]
    HC_random = HC_random[:sub_per_class]
    SZ_random = SZ_random[:sub_per_class]
    #

    # Choose the subject_per_class indices from HC_index_val and SZ_index_val
    # using random numbers

    HC_index_tr = total_HC_index_tr[HC_random]
    SZ_index_tr = total_SZ_index_tr[SZ_random]

    tr_index = torch.cat((HC_index_tr, SZ_index_tr))
    val_index = torch.cat((HC_index_val, SZ_index_val))
    test_index = torch.cat((HC_index_test, SZ_index_test))

    tr_index = tr_index.view(tr_index.size(0))
    val_index = val_index.view(val_index.size(0))
    test_index = test_index.view(test_index.size(0))

    train_featues = finalData2[tr_index, :, :, :]
    valid_features = finalData2[val_index, :, :, :]
    test_features = finalData2[test_index, :, :, :]

    train_labels = all_labels[tr_index.long()]
    valid_labels = all_labels[val_index.long()]
    test_labels = all_labels[test_index.long()]

    return (
        train_featues,
        train_labels,
        valid_features,
        valid_labels,
        test_features,
        test_labels,
    )
