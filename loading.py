import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from skimage.transform import resize
from visualisation import plot_df_samples
def load_db(folder_path, im_size, load, test):
    # List of all *.tab files in folder
    if load:
        file_names = [f for f in os.listdir(folder_path) if f.endswith('.hdf5')]
        if not file_names:
            raise ValueError("No .tab files found in the specified folder.")
        names, imgs, stps = [], [], []

        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)

            try:
                #members = h5py.File(file_path)['TRACE_DATA/DEFAULT']
                #for m in members:
                #    print(m)
                with h5py.File(file_path, 'r') as h5file:

                    def print_structure(name, obj):
                        print(f"{name}: {type(obj)}")
                        if isinstance(obj, h5py.Dataset):
                            print(f"  Shape: {obj.shape}")
                            print(f"  Dtype: {obj.dtype}")

                    # Visit each object in the file
                    #h5file.visititems(print_structure)
                    data_arr = h5file['TRACE_DATA/DEFAULT/data_array'][:]
                    samp_num = h5file['TRACE_DATA/DEFAULT/SAMP_NUM'][:]
                    rec_x = h5file['TRACE_DATA/DEFAULT/REC_X'][:]
                    rec_y = h5file['TRACE_DATA/DEFAULT/REC_Y'][:]
                    samp_rate = h5file['TRACE_DATA/DEFAULT/SAMP_RATE'][:]
                    f_break = h5file['TRACE_DATA/DEFAULT/SPARE1'][:]
                    pivots = []
                    pivots2 = []
                    for idx, x in enumerate(rec_x[:4000]):
                        if idx == 0:
                            continue
                        if x > rec_x[idx+1] and x > rec_x[idx-1]:
                            pivots.append(idx)
                        if x < rec_x[idx+1] and x < rec_x[idx-1]:
                            pivots2.append(idx)
                    data_splits = np.split(data_arr, pivots, axis=0)

                    break_splits = np.split(f_break, pivots, axis=0)
                    labels = [*range(len(break_splits))]
                    df = pd.DataFrame({'2d_array': data_splits, '1d_array': break_splits, 'Label': labels})
                    for idx, img in enumerate(df['2d_array']):
                        #resized_img = resize(img, (im_size, im_size), anti_aliasing=True)
                        normalized_img = img#(resized_img - resized_img.min()) / (resized_img.max() - resized_img.min())
                        df.at[idx, '2d_array'] = normalized_img
                        arr = df.loc[idx, '1d_array']
                        arr = arr.astype(object)
                        arr[arr == 0] = None
                        arr[arr == -1] = None
                        df.at[idx, '1d_array'] = arr
                        #df.loc[idx, '1d_array'][df.loc[idx, '1d_array'] == -1] = None

                    #df.to_hdf('data.h5', key='df', mode='w')
                    #arr1 = df['2d_array'].to_numpy()
                    #arr2 = df['1d_array'].to_numpy()
                    #np.savez('my_arrays.npz', arr1=arr1, arr2=arr2)
                    #pd.read_hdf('data.h5', key='df')
                    plot_df_samples(df)
                if data_splits is None:
                    raise Exception('cant read data file')

                if data_splits is None or data_splits is None:
                    raise Exception('cant create image from tab')
                print('loaded file with id ', file_name)

                if file_name is not None:
                    names.append(file_name)
                else:
                    raise ValueError(f" {file_name} tab is None")
            except Exception as e:
                print(f'Exception happened: Error loading file {file_name}: {e}')
                continue

        if file_name is None:
            raise ValueError("Failed to load db.")
        if not test:
            pass
            #names, imgs, stps = np.array(names), np.array(imgs), np.array(stps)
            #np.save(f'sample_ids_{im_size}.npy', names)
            #np.save(f'images_{im_size}.npy', imgs)
            #np.save(f'stps_{im_size}.npy', stps)

    else:
        if not os.path.exists('sample_ids.npy') or not os.path.exists('images.npy'):
            raise FileNotFoundError('Sample files not found. Set load=True to generate them.')

        names = np.load(f'sample_ids_{im_size}.npy')
        imgs = np.load(f'images_{im_size}.npy')
        stps = np.load(f'stps_{im_size}.npy')

        if names is None or imgs is None:
            raise ValueError("Failed to load signatures.")
    return data_splits




