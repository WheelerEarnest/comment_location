"""
Author: Wheeler Earnest

"""
import numpy as np
import pandas as pd
import torch

from collections import Counter
from torch.utils.data import Dataset

EMPTY_LINE = 'EMPTY'
COMMENT_ONLY_LINE = 'COMMENTEMPTY'


def read_comment_files(data_file, max_line_length):
    d_file = open(data_file, 'r')
    line_id = -1

    all_words = Counter()
    current_file_contents = file_of_code()

    # List containing the loc data of each file as a file_of_code object
    files_of_code = []

    current_file_id = ""

    for line in d_file:
        # Increment the line_id and pull data from the text file
        line_id += 1
        file_id, line_number, block_bnd, label, _, _, code, _, _ = line.strip().split("\t")

        if code == COMMENT_ONLY_LINE or EMPTY_LINE:
            continue

        # Clean up the code and count the words
        code_words = code.split()[0:min(len(code.split()), max_line_length)]
        for word in code_words:
            all_words[word] += 1
        code = " ".join(code_words).strip()

        current_loc = line_of_code(file_id, line_number, block_bnd, code, label)

        # Check to see if this loc is in the same file as the previous
        if file_id != current_file_id:
            if current_file_contents.len() > 0:
                files_of_code.append(current_file_contents)
                current_file_contents = file_of_code()
            current_file_id = file_id

        current_file_contents.add_loc(current_loc)

    if current_file_contents.len() > 0:
        files_of_code.append(current_file_contents)
    d_file.close()

    return files_of_code, all_words


class line_of_code(object):
    """
    Contains all the information about a specific line of code.
    """

    def __init__(self, file_id, line_number, block_bnd, code, label):
        """
        :param file_id: int | id number of the file the line is from
        :param line_number: int
        :param block_bnd: int | -1 if NA, 1 if start of code block, 2 if middle, and 3 if end
        :param code: string | words on the line of code
        :param label: int | 0/1, Tells if a acomment appears before this code
        """

        self.file_id = file_id
        self.line_number = line_number
        self.block_bnd = block_bnd
        self.code = code
        self.label = label


class file_of_code(object):
    """
    Contains all the lines of code for an individual file
    """

    def __init__(self):
        self.locs = []

    def add_loc(self, loc):
        self.locs.append(loc)

    def len(self):
        return len(self.locs)

    def get_loc(self, idx):
        return locs[idx]

