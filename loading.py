import os
import numpy as np
import h5py
import pandas as pd
from skimage.transform import resize
from visualisation import plot_df_samples
import time
from tqdm import tqdm

import memory_profiler
import gc

def valid(arr, i):
    if arr is not None \
            and len(arr) > i > 0 \
            and (arr[i] == 0 or arr[i] == -1) \
            and (arr[i+1] != 0 or arr[i+1] != -1)  \
            and (arr[i - 1] != 0 or arr[i - 1] != -1):
        return True
    else:
        return False

#@memory_profiler.profile
def load_db(folder_path, load, test, im_width, im_height):
    # List of all *.hdf5 files in folder
    if load:
        #traces_img, first_break_line = None, None
        file_names = [f for f in os.listdir(folder_path) if f.endswith('.hdf5')]
        if not file_names:
            raise ValueError("No .hdf5 files found in the specified folder.")

        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)


            with h5py.File(file_path, 'r') as h5file:
                data_arr = h5file['TRACE_DATA/DEFAULT/data_array'][:]
                samp_num = h5file['TRACE_DATA/DEFAULT/SAMP_NUM'][:]
                rec_x = h5file['TRACE_DATA/DEFAULT/REC_X'][:]
                #rec_y = h5file['TRACE_DATA/DEFAULT/REC_Y'][:]
                #samp_rate = h5file['TRACE_DATA/DEFAULT/SAMP_RATE'][:]
                f_break = h5file['TRACE_DATA/DEFAULT/SPARE1'][:]
                pivots = []
                pivots2 = []
                for idx, x in enumerate(rec_x[:]):
                    if idx == 0 or idx == len(rec_x)-1:
                        continue
                    if x > rec_x[idx+1] and x > rec_x[idx-1]:
                        pivots.append(idx)
                    #if x < rec_x[idx+1] and x < rec_x[idx-1]:
                        #pivots2.append(idx)
                traces_img = np.split(data_arr, pivots, axis=0)

                first_break_split = np.split(f_break, pivots, axis=0)
                first_break_lines = [arr.flatten().astype(float) for arr in first_break_split]

                n_traces = len(first_break_lines)
                labels = [*range(n_traces)]
                #df = pd.DataFrame({'2d_array': traces_img, '1d_array': first_break_lines, 'Label': labels})
                start_time = time.time()
                max_width_f = 0
                widths_f = []
                skipped = []
                for idx, arr in tqdm(enumerate(first_break_lines), total=len(first_break_lines),
                                     desc="Interpolating missing labels"):
                    n_lines = len(arr)
                    if n_lines == 0:
                        raise ValueError('empty first break line')
                    max_width_f = max(max_width_f, n_lines)
                    widths_f.append(n_lines)
                    arr[np.isin(arr, [0, -1])] = np.nan

                    mask = ~np.isnan(arr)
                    if mask.any():
                        x = np.arange(n_lines)
                        first_break_lines[idx] = np.interp(x, x[mask], arr[mask]).astype(int)
                        del x, mask
                    else:
                        skipped.append(idx)
                        print(f'skipped  line {idx}, all nans')
                        continue
                    if idx == n_lines-1 or idx == n_lines//2:
                        gc.collect()

                #plot_df_samples(df)
                end_time = time.time()
                print(f"Time taken to interpolate: {(end_time - start_time)/60:.1f} minutes")
                print(f"Skipped {len(skipped)} lines : {len(skipped)/n_lines:.1f} %")

                filtered_arr1 = [item for idx, item in enumerate(first_break_lines) if idx not in skipped]
                filtered_arr2 = [item for idx, item in enumerate(traces_img) if idx not in skipped]
                del first_break_lines, traces_img
                widths_t = []
                max_width_t = 0
                skipped = []
                start_time = time.time()
                for idx, img in tqdm(enumerate(filtered_arr2), total=len(filtered_arr2),
                                     desc=" Norm images"):
                    current_width = img.shape[0]
                    max_width_f = max(max_width_t, current_width)
                    widths_t.append(current_width)
                    if img.shape[0] < im_width:
                        padded_image = np.zeros((im_width, img.shape[1]))
                        padded_image[:img.shape[0], :] = img
                        img = padded_image
                        del padded_image
                    else:
                        img = img[:im_width, :]
                    #img = resize(img, (im_width, im_height), anti_aliasing=True)
                    if img.size > 0:
                        img_min = img.min()
                        img_max = img.max()
                    else:
                        raise ValueError("Image array is empty.")
                    if img_min == img_max:
                        raise ZeroDivisionError('Image data is Zero')
                    else:

                        img -= img_min
                        img /= (img_max - img_min)
                        filtered_arr2[idx] = img
                np.savez('my_arrays.npz', arr1=filtered_arr1, arr2=filtered_arr2, arr3=[max_width_f, max_width_t])
                end_time = time.time()
                print(f"Time taken to save: {(end_time - start_time) / 60:.1f} minutes")
                #df.to_hdf('data.h5', key='df', mode='w')

                if traces_img is None:
                    raise Exception('cant read data file')

                if traces_img is None or first_break_lines is None:
                    raise Exception('cant create image from trace')
                print('loaded file with id ', file_name)

                #if file_name is not None:
                #    names.append(file_name)
                #else:
                #    raise ValueError(f" {file_name} tab is None")
            #except Exception as e:
            #    print(f'Exception happened: Error loading file {file_name}: {e}')
            #    break

        if file_names is None:
            raise ValueError("Failed to load db.")
        if not test:
            pass
    else:
        if not os.path.exists('my_arrays.npz'):
            raise FileNotFoundError('Sample files not found. Set load=True to generate them.')
        loaded = np.load('my_arrays.npz', allow_pickle=True)
        traces_img = loaded['arr1']
        first_break_lines = loaded['arr2']
        max_width = loaded['arr3'][0]
        df = pd.DataFrame({'2d_array': traces_img, '1d_array': first_break_lines})
        #plot_df_samples(df)
        if traces_img is None or first_break_lines is None:
            raise ValueError("Failed to load traces.")
    return traces_img, first_break_lines, max_width




