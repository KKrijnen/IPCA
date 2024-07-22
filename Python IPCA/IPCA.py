from sklearn.decomposition import IncrementalPCA
import numpy as np
import math
import json 


def batch_est_fit(pixels, features):
    minimum_ratio = 20
    feature_offset = 20000 - 20 * features if features < 1000 else 0
    batch_size = features * 20 + feature_offset  # Initial Estimate
    trailing_batchsize = pixels % batch_size
    trailing_batchsize_ratio = trailing_batchsize // features

    while trailing_batchsize_ratio < minimum_ratio and trailing_batchsize != 0:
        number_of_loops = pixels // batch_size
        loop_upperbound = math.ceil(pixels / number_of_loops)
        loop_lowerbound = math.ceil(pixels // (number_of_loops + 1))
        trailing_treshold_difference = int((minimum_ratio * features - trailing_batchsize) / number_of_loops)

        if trailing_treshold_difference > 0 and (batch_size - trailing_treshold_difference) >= loop_lowerbound:
            batch_size -= trailing_treshold_difference
        else:
            batch_size = loop_upperbound

        if batch_size > 150000:
            batch_size = features * 21 + feature_offset  # Initial Estimate
            break

        if batch_size >= pixels:
            batch_size = pixels
            break
        
        trailing_batchsize = pixels % batch_size
        trailing_batchsize_ratio = trailing_batchsize / features
    normal_iterations = int(math.floor(pixels / batch_size))
    trailing_batchsize = pixels % batch_size
    batch_array = [batch_size for i in range(0, normal_iterations)]
    if trailing_batchsize != 0:
        batch_array.append(trailing_batchsize)

    return batch_array

def batch_est_transform(pixels, features):
    batch_size = math.ceil(20000000 / features)
    normal_iterations = int(math.floor(pixels / batch_size))
    trailing_batchsize = pixels % batch_size
    batch_array = [batch_size for i in range(0, normal_iterations)]
    if trailing_batchsize != 0:
        batch_array.append(trailing_batchsize)
    return batch_array

def MSI_IPCA(metadata, filepath):
    batch_size_fit = batch_est_fit(metadata["number_of_spectra"], metadata["number_of_peaks"])
    batch_size_transform = batch_est_transform(metadata["number_of_spectra"], metadata["number_of_peaks"])
    ipca = IncrementalPCA(n_components=metadata["number_of_peaks"])
    #Does a fit on a part of the data. The memory map is generated within the loop clear the ram. 
    #File buffering has been benchmarked and was slower than an memmap even when the memory map is being generated within the loop. 
    offset = 0
    for i in batch_size_fit:
        memory_map = np.memmap((filepath + ".bin"), dtype = 'float32', mode='r', shape=(metadata["number_of_spectra"], metadata["number_of_peaks"]))
        ipca.partial_fit(memory_map[offset:offset + i, :])
        offset += i
        del memory_map
    offset = 0
    #Transforms part of the data and writes it to the the output file    
    with open(filepath + "_results.bin", 'wb') as output_file:
        for i in batch_size_transform:
            memory_map = np.memmap((filepath + ".bin"), dtype = 'float32', mode='r', shape=(metadata["number_of_spectra"], metadata["number_of_peaks"]))
            ipca.transform(memory_map[offset:offset + i, :]).tofile(output_file, "") 
            offset += i
            del memory_map

def load_metadata(file_path):
    with open((file_path + ".json")) as json_file:
        return(json.load(json_file))