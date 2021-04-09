"""
Author: Wheeler Earnest

"""
import sys, os
import data_helper
from torch.utils.data import DataLoader

data_path = sys.argv[1]

if os.path.isdir(data_path):
    data_path = os.path.join(data_path, "data.txt")

if not os.path.exists(data_path):
    print("Please enter a valid path")

dataset = data_helper.create_dataset(data_path,  5000, 100, 100, 100, 100)

loader = DataLoader(dataset, batch_size=32)

print("Batch size 32")
for n_batch, (data, data_wts, labels, label_wts)  in enumerate(loader):
    print("Data: " +  str(data.size()) + "\nData Weights: " + str(data_wts.size()) + "\nLabels: " + str(labels.size()) +
          "\nLabel Weights: " + str(label_wts.size()))
    break

print("Batch size 64 ")
loader = DataLoader(dataset, batch_size=64)
for n_batch, (data, data_wts, labels, label_wts)  in enumerate(loader):
    print("Data: " +  str(data.size()) + "\nData Weights: " + str(data_wts.size()) + "\nLabels: " + str(labels.size()) +
          "\nLabel Weights: " + str(label_wts.size()))
    break