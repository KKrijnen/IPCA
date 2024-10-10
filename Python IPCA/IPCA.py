from sklearn.decomposition import IncrementalPCA
import numpy as np
import math
import json 

# Point II in modifications
#Attempts to find a batch size with a size to peak ratio of >19 including the final batch.
#If after the initial batch_size of 19 * peaks the size of the last batch that has a peak ratio <19. 
# This function will search for larger batch sizes that have a last batch that has a peak ratio >19.
def batch_est_fit(pixels, peaks):
    minimum_ratio = 19
    batch_size = peaks * minimum_ratio
    trailing_batchsize = pixels % batch_size
    trailing_batchsize_ratio = trailing_batchsize // peaks

    while trailing_batchsize_ratio < minimum_ratio and trailing_batchsize != 0:
        number_of_loops = pixels // batch_size
        loop_upperbound = math.ceil(pixels / number_of_loops)
        loop_lowerbound = math.ceil(pixels // (number_of_loops + 1))
        trailing_treshold_difference = int((minimum_ratio * peaks - trailing_batchsize) / number_of_loops) 

        if trailing_treshold_difference > 0 and (batch_size - trailing_treshold_difference) >= loop_lowerbound and ((batch_size - trailing_treshold_difference)/peaks) > minimum_ratio:
            batch_size -= trailing_treshold_difference
        else:
            batch_size = loop_upperbound

        if batch_size >= pixels:
            batch_size = pixels
            break
            
        
        trailing_batchsize = pixels % batch_size
        trailing_batchsize_ratio = trailing_batchsize / peaks
    normal_iterations = int(math.floor(pixels / batch_size))
    trailing_batchsize = pixels % batch_size
    batch_array = [batch_size for i in range(0, normal_iterations)]
    if trailing_batchsize != 0:
        batch_array.append(trailing_batchsize)
    return batch_array


#Keeps the total ammount that is loaded stable
#Loads 80MB of data at a time 
def batch_est_transform(pixels, peaks):
    batch_size = math.ceil(20000000 / peaks)
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
    #Point III in modifications. Iterates over the data in batches. Each batch will be is transformed and the results are saved to disk as a binary file. 
    with open(filepath + "_results.bin", 'wb') as output_file:
        for i in batch_size_transform:
            memory_map = np.memmap((filepath + ".bin"), dtype = 'float32', mode='r', shape=(metadata["number_of_spectra"], metadata["number_of_peaks"]))
            ipca.transform(memory_map[offset:offset + i, :]).tofile(output_file, "") 
            offset += i
            del memory_map
    return ipca

def load_metadata(file_path):
    with open((file_path + ".json")) as json_file:
        return(json.load(json_file))
