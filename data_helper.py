"""
Author: Wheeler Earnest

"""
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

EMPTY_LINE = 'EMPTY'
COMMENT_ONLY_LINE = 'COMMENTEMPTY'



def read_comment_files(data_file):

    d_file = open(data_file, 'r')
    line_id = -1
    file_ids = []
    line_nums = []
    block_bnds = []
    code_lines = []

    labels = []

    for line in d_file:
        #Increment the line_id and pull data from the text file
        line_id += 1
        file_id, line_number, block_bnd, label, _, _, code, _, _ = line.strip().split("\t")

        if code == COMMENT_ONLY_LINE or EMPTY_LINE:
            continue


