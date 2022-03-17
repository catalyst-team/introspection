import h5py
import pandas as pd
import numpy as np

# read the data
hf = h5py.File("COBRE_AllData.h5", "r")
data = hf.get("COBRE_dataset")
data = np.array(data)
num_subjects = data.shape[0]
num_components = 100
data = data.reshape(num_subjects, num_components, -1)
print("data shape: ", data.shape)

# take only those brain networks that are not noise
filename = "correct_indices_GSP.csv"
df = pd.read_csv(filename, header=None)
c_indices = df.values
c_indices = c_indices.astype("int")
c_indices = c_indices.flatten()
c_indices = c_indices - 1
finalData = data[:, c_indices, :]
print("data shape: ", finalData.shape)

filename = "labels_COBRE.csv"
df = pd.read_csv(filename, header=None)
labels = df.values.flatten() - 1
