import os
from cTab import CTab
import numpy as np
from C_Truth import C_Truth
from tabToImage import tabToImage


def load_signatures(folder_path, im_size, load, test):
    # List of all *.tab files in folder
    if load:
        files = [f for f in os.listdir(folder_path) if f.endswith('.tab')]
        if not files:
            raise ValueError("No .tab files found in the specified folder.")
        names, imgs, stps = [], [], []

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            try:
                tab = CTab(file_path)
                if tab is None:
                    raise Exception('cant read tab file')
                img, stp = tabToImage(tab, im_size)
                if img is None or stp is None:
                    raise Exception('cant create image from tab')
                print('loaded file with id ', file_name)
                imgs.append(img)
                stps.append(stp)
                if file_name is not None:
                    names.append(file_name)
                else:
                    raise ValueError(f" {file_name} tab is None")
            except Exception as e:
                print(f'Error loading file {file_name}: {e}')
                continue

        if not names or imgs is None:
            raise ValueError("Failed to load signatures.")
        if not test:
            #names, imgs, stps = np.array(names), np.array(imgs), np.array(stps)
            np.save(f'sample_ids_{im_size}.npy', names)
            np.save(f'images_{im_size}.npy', imgs)
            np.save(f'stps_{im_size}.npy', stps)

    else:
        if not os.path.exists('sample_ids.npy') or not os.path.exists('images.npy'):
            raise FileNotFoundError('Sample files not found. Set load=True to generate them.')

        names = np.load(f'sample_ids_{im_size}.npy')
        imgs = np.load(f'images_{im_size}.npy')
        stps = np.load(f'stps_{im_size}.npy')

        if names is None or imgs is None:
            raise ValueError("Failed to load signatures.")
    return names, imgs, stps


def load_form_and_truth(sample_ids):

    c_truth = C_Truth()
    forms, truth = [], []
    if c_truth.load("images/truth.txt"):
        for s_id in sample_ids:
            try:
                truth_line = c_truth.get(s_id, 100)
                if truth_line:
                    parts = truth_line.split(' ')
                    if len(parts) != 3:
                        raise ValueError(f"Expected 3 elements in truth line, got {len(parts)}: {truth_line}")

                    _, form, isgenuine = truth_line.split(' ')
                    if form is not None:
                        forms.append(form)
                    else:
                        raise Exception('error reading form')
                    if isgenuine is None:
                        raise Exception('error reading genuine/forged')
                    elif isgenuine == 'genuine':
                        truth.append(1)
                    elif isgenuine == 'forged':
                        truth.append(0)
                    else:
                        raise Exception(f'signature verdict must be genuine or forged, found {isgenuine}')
                else:
                    print("File not found in dictionary")
                    return
            except Exception as e:
                print(f'error occurred, {e}')

        return forms, truth
    else:
        print('truth file loading error')
        return



