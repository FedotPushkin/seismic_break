import os
import gc
import h5py
import time
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from visualisation import plot_train_samples
from visualisation import show_mask_samples, plot_train_sample
import memory_profiler
from build_data import create_tf_dataset_from_hdf5
from utils import compose_filters,remove_outliers_z_score,remove_outliers_iqr,remove_outliers_moving_average


def get_pivots(array):
    pivots = []

    for idx, x in enumerate(array[:]):
        if idx == 0 or idx == len(array) - 1:
            continue
        if x > array[idx + 1] and x > array[idx - 1]:
            pivots.append(idx)
    return pivots


def create_masks(filtered_arr1, max_width, train_shape, samp_rate, ds_height):
    masks = []
    coef = (1000 / samp_rate).astype(np.float32)
    coef_y = train_shape[1] / ds_height
    for idx, arr in tqdm(enumerate(filtered_arr1), desc="Creating masks"):
        if arr.shape[0] == 0 or arr.shape[0] > max_width or (np.array(arr) < 0).any():
            raise ValueError('array larger then max_width')
        elif 0 < arr.shape[0] <= max_width:

            max_y = max(arr)
            min_y = min(arr)
            if min_y != max_y:

                y_line_norm = (arr - min_y) / (max_y - min_y)
                new_len = round(arr.shape[0] * train_shape[0] / max_width)
                num_zeros_resized = train_shape[0] - new_len

                y_line_resized = resize(y_line_norm, (new_len,), anti_aliasing=True)

                max_yr = max(y_line_resized)
                min_yr = min(y_line_resized)

                y_line_resized = (y_line_resized - min_yr) / (max_yr - min_yr)
                y_line_resized = y_line_resized * (max_y - min_y) + min_y

                y_line_resized = np.concatenate((y_line_resized,
                                                 np.zeros(num_zeros_resized, dtype=np.float32)))
                filtered_arr1[idx] = (y_line_resized * coef * coef_y).astype(np.int32)
                del y_line_resized
            else:

                if len(arr) <= train_shape[0]:
                    num_zeros_resized = max(train_shape[0] - len(arr), 0)
                    filtered_arr1[idx] = np.round(
                        np.concatenate((arr * coef * coef_y, np.zeros(num_zeros_resized, dtype=np.float32)))
                        ).astype(np.int32)
                else:
                    newarr = arr[:train_shape[0]] * coef * coef_y
                    filtered_arr1[idx] = newarr.astype(np.int32)

        else:
            raise ValueError('arr longer then max_width_f')

        min_yr = min(filtered_arr1[idx])
        if min_yr < 0:
            raise ValueError('negative ')
        y_line_resized = filtered_arr1[idx]
        mask = np.zeros(train_shape, dtype=np.float32)
        if len(y_line_resized) < 1 or len(y_line_resized) > train_shape[0]:
            raise ValueError('mask array out of range')
        for x in range(train_shape[0]):
            y_of_train_shape = y_line_resized[x]

            if 0 < y_of_train_shape < train_shape[1]:
                mask[x, y_of_train_shape] = 1
                mask[x, y_of_train_shape + 1:] = 2
            elif y_of_train_shape == 0:
                continue
            elif y_of_train_shape >= train_shape[1]:
                raise ValueError('mask array out of range')

        masks.append(mask)
        if len(masks) == 6 and 0:
            show_mask_samples(masks)
        del mask
    return masks


def split_data(pivots, not_splitted_first_breaks):
    max_width_f = 0
    skipped = []
    first_break_split = np.split(not_splitted_first_breaks, pivots, axis=0)

    first_break_lines = [arr.flatten().astype(float) for arr in first_break_split]

    for idx, arr in tqdm(enumerate(first_break_lines), total=len(first_break_lines),
                         desc="Interpolating missing labels"):
        n_lines = len(arr)
        if n_lines == 0:
            raise ValueError('empty first break line')
        max_width_f = max(max_width_f, n_lines)
        arr[np.isin(arr, [0, -1])] = np.nan

        mask = ~np.isnan(arr)
        if mask.any() and max(arr) != min(arr):
            x = np.arange(n_lines)
            first_break_lines[idx] = np.interp(x, x[mask], arr[mask]).astype(int)
            first_break_lines[idx] = remove_outliers_z_score(first_break_lines[idx])
            first_break_lines[idx] = remove_outliers_moving_average(first_break_lines[idx], window_size=5, threshold=2)
            first_break_lines[idx] = np.nan_to_num(first_break_lines[idx], nan=0)
            del x, mask
        else:

            skipped.append(idx)
            continue
        if idx == n_lines - 1 or idx == n_lines // 2:
            gc.collect()

    for idx, arr in tqdm(enumerate(first_break_lines), total=len(first_break_lines),
                         desc="Skpping data labels less then 1/3 of max_width"):
        if arr is None:
            skipped.append(idx)
            continue
        elif len(arr) < max_width_f // 3:
            skipped.append(idx)
    return first_break_lines, skipped, max_width_f


def norm_images(filtered_arr2, max_width, train_shape):
    for idx, img in tqdm(enumerate(filtered_arr2), total=len(filtered_arr2),
                         desc=" Norm images"):
        current_width = img.shape[0]
        if max_width < current_width:
            raise ValueError('img shape error')

        new_width = int(round((train_shape[0] * current_width / max_width)))
        img = resize(img, (new_width, train_shape[1]), anti_aliasing=True)
        if img.shape[0] < max_width:
            padded_image = np.zeros(train_shape, dtype=np.float32)
            padded_image[:img.shape[0], :] = img
            img = padded_image
            del padded_image
        else:
            img = img[:max_width, :]
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
    return filtered_arr2


def load_db(args):

    train_shape = args.train_shape
    chunk_size = args.chunk_size
    # List of all *.hdf5 files in folder
    if args.load:

        file_names = [f for f in os.listdir(args.folder) if (f.endswith('.hdf5') or (f.endswith('.hdf')))]
        if not file_names:
            raise ValueError("No .hdf5 files found in the specified folder.")

        for file_name in file_names:
            file_path = os.path.join(args.folder, file_name)

            with h5py.File(file_path, 'r') as h5file:
                data_length = h5file['TRACE_DATA/DEFAULT/data_array'].shape[0]
                for i in range(0, data_length, chunk_size):

                    print(f'loading file {file_name} chunk {1+i//chunk_size} of {1+data_length//chunk_size}')
                    start_time = time.time()
                    data_arr = h5file['TRACE_DATA/DEFAULT/data_array'][i:i + chunk_size]
                    samp_num_arr = h5file['TRACE_DATA/DEFAULT/SAMP_NUM'][i:i + chunk_size]
                    rec_x = h5file['TRACE_DATA/DEFAULT/REC_X'][i:i + chunk_size]
                    samp_rate_arr = h5file['TRACE_DATA/DEFAULT/SAMP_RATE'][i:i + chunk_size]
                    not_splitted_first_breaks = h5file['TRACE_DATA/DEFAULT/SPARE1'][i:i + chunk_size]
                    samp_rate_arr = np.array(samp_rate_arr).flatten()
                    samp_num_arr = np.array(samp_num_arr).flatten()

                    if not np.all(samp_rate_arr == samp_rate_arr[0]):
                        raise ValueError('samp rate not constant')
                    if not np.all(samp_num_arr == samp_num_arr[0]):
                        raise ValueError('samp num not constant')

                    samp_rate = samp_rate_arr[0]
                    ds_height = samp_num_arr[0]
                    if args.plot_samples:
                        plot_train_sample(data_arr[:371], not_splitted_first_breaks[:371]*1000/samp_rate)
                    pivots = get_pivots(rec_x)
                    first_break_lines, skipped, max_width = split_data(pivots, not_splitted_first_breaks)

                    print(f"Skipped {len(skipped)} lines : {len(skipped)/len(first_break_lines):.1f} %")

                    filtered_arr1 = [item for idx, item in enumerate(first_break_lines) if idx not in skipped]
                    traces_img = np.split(data_arr, pivots, axis=0)
                    del not_splitted_first_breaks, data_arr, rec_x, samp_rate_arr, samp_num_arr
                    print('Applying filters to images')
                    filtered_arr2 = [item for idx, item in enumerate(traces_img) if idx not in skipped]
                    for idx, item in tqdm(enumerate(filtered_arr2), desc='Applying filters to images'):
                        filtered_arr2[idx] = compose_filters(item, samp_rate, args)

                    if len(filtered_arr1) == 0 or len(filtered_arr2) == 0:
                        print('Blank input array detected')
                    for idx, arr in tqdm(enumerate(filtered_arr1), desc="Cleaning data"):
                        clean_data(arr, filtered_arr1[idx])

                    if len(filtered_arr1) > 0 and len(filtered_arr2) > 0:
                        plot_train_sample(filtered_arr2[0], filtered_arr1[0] * 0.5)

                    masks = create_masks(filtered_arr1, max_width, train_shape, samp_rate, ds_height)
                    filtered_arr2 = norm_images(filtered_arr2, max_width, train_shape)

                    traces_img = np.reshape(filtered_arr2, (len(filtered_arr2), train_shape[0], train_shape[1], 1))
                    masks = np.reshape(masks, (len(masks), train_shape[0], train_shape[1], 1))

                    del first_break_lines, filtered_arr2
                    gc.collect()
                    if args.plot_samples:
                        plot_train_samples(traces_img[:50], masks[:50], train_shape)
                        plot_train_samples(traces_img[6:50], masks[6:50], train_shape)
                        plot_train_samples(traces_img[12:50], masks[12:50], train_shape)
                        plot_train_samples(traces_img[18:50], masks[18:50], train_shape)
                        plot_train_samples(traces_img[24:50], masks[24:50], train_shape)
                        plot_train_samples(traces_img[30:50], masks[30:50], train_shape)
                        plot_train_samples(traces_img[36:50], masks[36:50], train_shape)
                        plot_train_samples(traces_img[42:50], masks[42:50], train_shape)
                        plot_train_samples(traces_img[48:54], masks[48:54], train_shape)
                        plot_train_samples(traces_img[54:60], masks[54:60], train_shape)
                    print('Saving to array, this will take a while')
                    if masks is None or traces_img is None:
                        raise Exception('cant create image from trace')
                    drop_train_data_to_file(masks=masks, traces_img=traces_img, train_shape=train_shape)
                    del masks, traces_img
                    end_time = time.time()
                    print(f"Time taken to load: {(end_time - start_time) / 60:.1f} minutes")
                    print(f'loaded file with id { file_name} chunk_{i//chunk_size}')


                #except Exception as e:
                #    print(f'Exception happened: Error loading file {file_name}: {e}')
                #    break

        if file_names is None:
            raise ValueError("Failed to load db.")

    if not os.path.exists('train_dataset.hdf5'):
        raise FileNotFoundError('Sample files not found. Set load=True to generate them.')
    dataset, val_dataset, train_samples, test_samples = \
        create_tf_dataset_from_hdf5('train_dataset.hdf5',
                                    batch_size=args.batch_size,
                                    chunk_size=chunk_size,
                                    train_ratio=0.8,
                                    train_shape=train_shape,
                                    plot_samples=args.plot_samples)

    return dataset, val_dataset, train_samples, test_samples


def clean_data(line, img):
    if line is None:
        raise ValueError('line None')
    if img is None:
        raise ValueError('img None')

    if len(line) == 0 or img.shape[0] == 0:
        print('empty line or empty img to be cleaned??')
        raise
    if len(line) != img.shape[0]:
        raise ValueError(f'shape mismatch: img shape {img.shape}, line {line.shape}')

    for i, value in enumerate(line):
        if i < 3:
            continue

        if value != 0 and value == line[i-1] and value == line[i-2] and value == line[i-3]:
            idx = i-2
            while idx < len(line):
                if line[idx-1] == line[idx]:
                    line[idx-1] = 0
                    idx += 1
                else:
                    break
    zero_indices = [index for index, value in enumerate(line) if value == 0]
    img[zero_indices, :] = np.zeros(img.shape[1], dtype=np.float32)


def drop_train_data_to_file(masks, traces_img, train_shape):

    with h5py.File('train_dataset.hdf5', 'a') as h5file:

        # Check if the datasets already exist
        if 'masks' in h5file:
            # If exists, resize the dataset and add new data
            h5file['masks'].resize((h5file['masks'].shape[0] + masks.shape[0]), axis=0)
            h5file['masks'][-masks.shape[0]:] = masks  # Add new data at the end
        else:
            # Create the dataset if it doesn't exist
            h5file.create_dataset('masks', data=masks,
                                  maxshape=(None, train_shape[0], train_shape[1], 1), chunks=True)

        if 'traces_img' in h5file:
            # If exists, resize the dataset and add new data
            h5file['traces_img'].resize((h5file['traces_img'].shape[0] + traces_img.shape[0]), axis=0)
            h5file['traces_img'][-traces_img.shape[0]:] = traces_img  # Add new data at the end
        else:
            # Create the dataset if it doesn't exist
            h5file.create_dataset('traces_img', data=traces_img,
                                  maxshape=(None, train_shape[0], train_shape[1], 1), chunks=True)

    print("New arrays added to HDF5 file successfully.")

