import numpy as np


#################################################################################
# %% distance
#################################################################################
def distance(f, w, value = 0):
    if value == 0:
        f_expand = np.expand_dims(np.expand_dims(f, axis=1), axis=1)
        w_expand = np.expand_dims(w, axis=1)
        fw = f_expand - w_expand
        dist = fw ** 2
        dist = (np.sum(np.squeeze(dist), -1)) ** 0.5
    else:
        f_expand = np.expand_dims(np.expand_dims(f, axis=1), axis=1)
        w_expand = np.expand_dims(w, axis=1)
        fw = f_expand - w_expand
        dist = fw ** 2
        dist = (np.sum(np.squeeze(dist), -1)) ** 0.5

    return dist
