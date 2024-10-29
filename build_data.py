import pandas as pd
import itertools
import numpy as np
import tensorflow as tf


def create_fit_data(sample_ids, images, stps, forms, truth, im_size):
    if len(sample_ids) == 0 or len(images) == 0 or len(forms) == 0 or len(truth) == 0:
        raise ValueError("Input lists must not be empty.")
    if len(sample_ids) == len(images) == len(forms) == len(truth):

        images = np.reshape(images, (images.shape[0], im_size, im_size, 1))
        stps = np.reshape(np.array(stps), (stps.shape[0], im_size, im_size, 1))
        img_combined = tf.concat([images, stps], axis=-1)
        df = pd.DataFrame(
            {'id': sample_ids,
             'form': forms,
             'true': truth
             })
        unique_forms = list(set(forms))
        pos_examples, neg_examples = [], []
        for form in unique_forms:
            gids = df[(df['form'] == form) & (df['true'] == 1)]['id'].tolist()
            fids = df[(df['form'] == form) & (df['true'] == 0)]['id'].tolist()
            if len(gids) > 1:
                pos_examples += list(itertools.permutations(gids, 2))

            for g in gids:
                for f in fids:
                    neg_examples.append((g, f))
                    neg_examples.append((f, g))

        imdict = dict(zip(sample_ids, img_combined))
        y = np.concatenate([np.ones(len(pos_examples)), np.zeros(len(neg_examples))])

        Xt1, Xt2 = [], []
        for p in pos_examples:
            Xt1.append(imdict[p[0]])
            Xt2.append(imdict[p[1]])
        for n in neg_examples:
            Xt1.append(imdict[n[0]])
            Xt2.append(imdict[n[1]])

        Xt1 = np.array(Xt1)
        Xt2 = np.array(Xt2)
        y = np.array(y)
    else:
        raise ValueError("Length of all input lists must be the same.")
    return Xt1, Xt2, y


def create_test_data(images, stps, ev_image, ev_stps, im_size):
    if len(images) == 0 or len(ev_image) == 0:
        raise ValueError("Input lists must not be empty.")
    if len(ev_image) == 1:
        Xt1, Xt2 = [], []
        images = np.reshape(images, (len(images), im_size, im_size, 1))
        stps = np.reshape(np.array(stps), (len(stps), im_size, im_size, 1))
        img_combined = tf.concat([images, stps], axis=-1)
        ev_image = np.reshape(ev_image, (len(ev_image), im_size, im_size, 1))
        ev_stps = np.reshape(np.array(ev_stps), (len(ev_stps), im_size, im_size, 1))
        ev_combined = tf.concat([ev_image, ev_stps], axis=-1)
        for img in img_combined:
            Xt1.append(img)
            Xt1.append(ev_combined[0])
            Xt2.append(ev_combined[0])
            Xt2.append(img)

        return np.array(Xt1), np.array(Xt2)
    else:
        raise ValueError("Length of eval must be 1.")
