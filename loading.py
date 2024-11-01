import os
import numpy as np
import h5py
from skimage.transform import resize
from visualisation import plot_train_samples
import time
from tqdm import tqdm
from visualisation import show_mask_samples,show2withaxis,plot_train_sample
import memory_profiler
import gc
import math


#@memory_profiler.profile

def resize_array(arr, new_length):
    original_indices = np.linspace(0, 1, len(arr))
    new_indices = np.linspace(0, 1, new_length)
    return np.interp(new_indices, original_indices, arr)
def load_db(folder_path, train_shape, load, test):
    # List of all *.hdf5 files in folder
    if load:

        file_names = [f for f in os.listdir(folder_path) if f.endswith('.hdf5')]
        if not file_names:
            raise ValueError("No .hdf5 files found in the specified folder.")

        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)

            with h5py.File(file_path, 'r') as h5file:
                data_arr = h5file['TRACE_DATA/DEFAULT/data_array'][:]
                samp_num_arr = h5file['TRACE_DATA/DEFAULT/SAMP_NUM'][:]
                rec_x = h5file['TRACE_DATA/DEFAULT/REC_X'][:]
                #rec_y = h5file['TRACE_DATA/DEFAULT/REC_Y'][:]
                samp_rate_arr = h5file['TRACE_DATA/DEFAULT/SAMP_RATE'][:]
                f_break = h5file['TRACE_DATA/DEFAULT/SPARE1'][:]
                samp_rate_arr = np.array(samp_rate_arr).flatten()
                samp_num_arr = np.array(samp_num_arr).flatten()

                #plot_train_sample(data_arr[:171], f_break[:171]*0.5)
                if not np.all(samp_rate_arr == samp_rate_arr[0]):
                    raise ValueError('samp rate not constant')
                if not np.all(samp_num_arr == samp_num_arr[0]):
                    raise ValueError('samp num not constant')
                samp_rate = samp_rate_arr[0]

                ds_height = samp_num_arr[0]
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
                del f_break, data_arr, rec_x, samp_rate_arr, samp_num_arr
                #n_traces = len(first_break_lines)

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
                    #widths_f.append(n_lines)
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
                shapes = []
                filtered_arr1 = [item for idx, item in enumerate(first_break_lines) if idx not in skipped]

                filtered_arr2 = [item for idx, item in enumerate(traces_img) if idx not in skipped]

                coef = (1000 / samp_rate).astype(np.float64)
                coef_y = train_shape[1] / ds_height
                masks = []
                #plot_train_sample(filtered_arr2[0], filtered_arr1[0] * 0.5)

                for idx, arr in tqdm(enumerate(filtered_arr1), desc="Creating masks"):
                    if arr.shape[0] == 0 or arr.shape[0] > max_width_f:
                        raise ValueError('array larger then max_width')
                    elif 0 < arr.shape[0] <= max_width_f:

                        max_y = max(arr)
                        min_y = min(arr)
                        if min_y != max_y:

                            y_line_norm = (arr-min_y) / (max_y-min_y)
                            new_len = round(arr.shape[0]*train_shape[0]/max_width_f)
                            num_zeros_resized = train_shape[0] - new_len

                            y_line_resized = resize(y_line_norm, (new_len,), anti_aliasing=True)

                            max_yr = max(y_line_resized)
                            min_yr = min(y_line_resized)

                            y_line_resized = (y_line_resized - min_yr)/(max_yr - min_yr)
                            y_line_resized = y_line_resized * (max_y - min_y) + min_y

                            y_line_resized = np.concatenate((y_line_resized, np.zeros(num_zeros_resized, dtype=np.float32)))
                            filtered_arr1[idx] = (y_line_resized * coef * coef_y).astype(int)
                        else:

                            if len(arr) <= train_shape[0]:
                                num_zeros_resized = train_shape[0] - len(arr)
                                filtered_arr1[idx] = (np.concatenate((arr, np.zeros(num_zeros_resized, dtype=np.float32))) * coef * coef_y).astype(int)
                            else:
                                filtered_arr1[idx] = (arr[:train_shape[0]] * coef * coef_y).astype(int)
                                #raise ValueError('arr longer then max_width_f')
                    else:
                        raise ValueError('arr longer then max_width_f')

                    #max_yr = max(y_line_resized)
                    min_yr = min(y_line_resized)
                    if min_yr < 0:
                        raise ValueError('negative ')
                    #print(min_yr, max_yr)
                    y_line_resized = filtered_arr1[idx]
                    mask = np.zeros(train_shape, dtype=np.float32)
                    if len(y_line_resized) < 1 or len(y_line_resized) > train_shape[0]:
                        raise ValueError('mask array out of range')
                    for x in range(train_shape[0]):
                        y_of_train_shape = y_line_resized[x]
                        # Calculate y value on the line for this x
                        if 0 < y_of_train_shape < train_shape[1]:
                            mask[x, y_of_train_shape] = 1
                            #mask[x, 0:y_train_shape] = 0
                            mask[x, y_of_train_shape + 1:] = 2
                        elif y_of_train_shape == 0:
                            continue
                        elif y_of_train_shape >= train_shape[1]:
                            raise ValueError('mask array out of range')

                    masks.append(mask)
                    if len(masks) == 6:
                        show_mask_samples(masks, [*range(6)])
                    del mask


                start_time = time.time()

                for idx, img in tqdm(enumerate(filtered_arr2), total=len(filtered_arr2),
                                     desc=" Norm images"):
                    current_width = img.shape[0]
                    if max_width_f < current_width:
                        raise ValueError('img shape error')
                    #widths_t.append(current_width)
                    new_width = int(round((train_shape[0]*current_width/max_width_f)))
                    img = resize(img, (new_width, train_shape[1]), anti_aliasing=True)
                    if img.shape[0] < max_width_f:
                        padded_image = np.zeros(train_shape, dtype=np.float32)
                        padded_image[:img.shape[0], :] = img
                        img = padded_image
                        del padded_image
                    else:
                        img = img[:max_width_f, :]
                    if img.shape != train_shape:
                        raise ValueError('img size unexpected')

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
                shape_arr0, shape_arr1 = [], []
                for f in filtered_arr2:
                    shape_arr0.append(f.shape[0])
                    shape_arr1.append(f.shape[1])
                print('before reshape', max(shape_arr0), min(shape_arr0), np.mean(shape_arr0))
                print(max(shape_arr1), min(shape_arr1), np.mean(shape_arr1))
                traces_img = np.reshape(filtered_arr2, (len(filtered_arr2), train_shape[0], train_shape[1], 1))
                masks = np.reshape(masks, (len(masks), train_shape[0], train_shape[1], 1))
                shape_arr0, shape_arr1 = [], []
                for f in masks:
                    shape_arr0.append(f.shape[0])
                    shape_arr1.append(f.shape[1])
                print('after reshape',max(shape_arr0), min(shape_arr0), np.mean(shape_arr0))
                print(max(shape_arr1), min(shape_arr1), np.mean(shape_arr1))
                del first_break_lines, filtered_arr2
                gc.collect()
                plot_train_samples(traces_img[:36], masks[:36], train_shape)
                print('saving to array, this will take a while')
                if masks is None or traces_img is None:
                    raise Exception('cant create image from trace')
                np.savez('my_arrays.npz', arr1=masks, arr2=traces_img)
                end_time = time.time()
                print(f"Time taken to save: {(end_time - start_time) / 60:.1f} minutes")
                #df.to_hdf('data.h5', key='df', mode='w')

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
        masks = loaded['arr1']
        traces_img = loaded['arr2']
        plot_train_samples(traces_img[:36], masks[:36], train_shape)
        if traces_img is None or masks is None:
            raise ValueError("Failed to load traces.")
    return traces_img, masks




