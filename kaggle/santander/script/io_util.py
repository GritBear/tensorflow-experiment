# Input parser for Airbnb data in csv
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import csv

PATH = '../data/'

def write_csv(filename, data):
    print(data)
    
    with open(PATH + filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for x in range(data.shape[0]): 
            #for x in range(data.size):
            # debug
            writer.writerow([data[x]]) # it is 1D array, thus no need to add [x, :].tolist()

    print('CSV data written' + PATH + filename)

    
def extract_txt_arr(filename, valueType=float, delimiter=',', hasHeader = True):
    with open(PATH + filename, 'rb') as f:
        headerRow = []
        data = []
        for line in f:
            strSplit = line.split(delimiter)
            if hasHeader and not headerRow:
                headerRow = strSplit
            else:   
                results = map(valueType, strSplit)
                data.append(results)
        return np.array(data), headerRow