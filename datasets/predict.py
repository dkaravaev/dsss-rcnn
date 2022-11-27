import numpy as np


def oos_predict(model, X, i, steps=20):
    N = X[i].shape[0]
    ret = X[i]
    for step in range(steps):
        p = model.predict(ret[np.newaxis, step:])
        if step == 0:
            ret = np.vstack((ret[0], p[0]))
        else:
            ret = np.vstack((ret, p[0, -1, :]))
    return ret


def ins_predict(model, X, i, steps=20):
    p = model.predict(X[i : steps + i])
    return p[:, -1, :]