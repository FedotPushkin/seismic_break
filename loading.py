import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from skimage.transform import resize
from visualisation import plot_df_samples
import time

def load_db(folder_path, im_size, load, test):
    # List of all *.tab files in folder
    if load:
        traces_img, first_break_line = None, None
        file_names = [f for f in os.listdir(folder_path) if f.endswith('.hdf5')]
        if not file_names:
            raise ValueError("No .tab files found in the specified folder.")
        names, imgs, stps = [], [], []

        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            try:

                with h5py.File(file_path, 'r') as h5file:
                    data_arr = h5file['TRACE_DATA/DEFAULT/data_array'][:]
                    samp_num = h5file['TRACE_DATA/DEFAULT/SAMP_NUM'][:]
                    rec_x = h5file['TRACE_DATA/DEFAULT/REC_X'][:]
                    rec_y = h5file['TRACE_DATA/DEFAULT/REC_Y'][:]
                    samp_rate = h5file['TRACE_DATA/DEFAULT/SAMP_RATE'][:]
                    f_break = h5file['TRACE_DATA/DEFAULT/SPARE1'][:]
                    pivots = []
                    pivots2 = []
                    for idx, x in enumerate(rec_x):
                        if idx == 0 or idx == len(rec_x)-1:
                            continue
                        if x > rec_x[idx+1] and x > rec_x[idx-1]:
                            pivots.append(idx)
                        #if x < rec_x[idx+1] and x < rec_x[idx-1]:
                            #pivots2.append(idx)
                    traces_img = np.split(data_arr, pivots, axis=0)

                    first_break_line = np.split(f_break, pivots, axis=0)
                    n_traces = len(first_break_line)
                    labels = [*range(n_traces)]
                    df = pd.DataFrame({'2d_array': traces_img, '1d_array': first_break_line, 'Label': labels})
                    start_time = time.time()
                    for idx, img in enumerate(df['2d_array']):
                        resized_img = resize(img, (im_size, im_size), anti_aliasing=True)
                        normalized_img = (resized_img - resized_img.min()) / (resized_img.max() - resized_img.min())
                        df.at[idx, '2d_array'] = normalized_img
                        arr = df.loc[idx, '1d_array']
                        arr = arr.astype(object)
                        arr[arr == 0] = None
                        arr[arr == -1] = None
                        df.at[idx, '1d_array'] = arr
                        if idx % 300 == 0:
                            print(f'loaded {idx} traces of {n_traces}')
                        #df.loc[idx, '1d_array'][df.loc[idx, '1d_array'] == -1] = None
                    end_time = time.time()
                    print(f"Time taken to resize&norm: {(end_time - start_time)/60} minutes")
                    start_time = time.time()
                    arr1 = df['2d_array'].to_numpy()
                    arr2 = df['1d_array'].to_numpy()
                    np.savez('my_arrays.npz', arr1=arr1, arr2=arr2)
                    end_time = time.time()
                    print(f"Time taken to save: {(end_time - start_time) / 60} minutes")
                    #df.to_hdf('data.h5', key='df', mode='w')
                    #plot_df_samples(df)
                if traces_img is None:
                    raise Exception('cant read data file')

                if traces_img is None or first_break_line is None:
                    raise Exception('cant create image from trace')
                print('loaded file with id ', file_name)

                if file_name is not None:
                    names.append(file_name)
                else:
                    raise ValueError(f" {file_name} tab is None")
            except Exception as e:
                print(f'Exception happened: Error loading file {file_name}: {e}')
                continue

        if file_names is None:
            raise ValueError("Failed to load db.")
        if not test:
            pass
    else:
        if not os.path.exists('my_arrays.npz'):
            raise FileNotFoundError('Sample files not found. Set load=True to generate them.')
        loaded = np.load('my_arrays.npz')
        traces_img = loaded[0]
        first_break_line = loaded[1]

        if traces_img is None or first_break_line is None:
            raise ValueError("Failed to load traces.")
    return traces_img, first_break_line




