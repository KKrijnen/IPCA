import numpy as np
import json
import math 
import time
import struct
import IPCA as ipca 
import os
from tkinter import filedialog
import matplotlib.pyplot as plt

# Loading the .json takes quite a bit of RAM in python for datasets.
# It is possible to remove the pixel information from the .json as
# this is not used for the IPCA itself, which is advicable for datasets
# containing a lot of pixels. Consider saving the pixel data in a different binary.
# for generating images. 


file_path = filedialog.askopenfilename(filetypes=[("json", "*.json")])[:-5]

print("Loading metadata\n")
metadata = ipca.load_metadata(file_path)
print("Metadata loaded, starting IPCA\n")
start_time = time.time()
ipca.MSI_IPCA(metadata, file_path)
end_time = time.time()
execution_time = end_time - start_time
print(f"IPCA finished, execution time: {execution_time} seconds\n")

# The transformed data will be saved to an 32-bit float binary, the same
# size as the original dataset. A memory map can be used to work with it.
# Example code can be of making an image can be seen below. 
#
# Does not work well on very large datasets (~256.000.000 pixels).


# component = 0

# results = np.memmap((file_path + "_results.bin"), 
#                     dtype = 'float32', mode='r',
#                     shape=(metadata["number_of_spectra"], metadata["number_of_peaks"]))
# im_shape = (metadata["region"][metadata["region_names"][0]]["x_max"]+1,metadata["region"][metadata["region_names"][0]]["y_max"]+1)
# im_mat = np.empty(im_shape)

# for spectrum_index, pixel in zip(range(0,metadata["number_of_spectra"]), metadata["region"][metadata["region_names"][0]]["pixels"]):
#     im_mat[pixel["x"], pixel["y"]] = results[spectrum_index, component]

# plt.imshow(im_mat)
# plt.show()
