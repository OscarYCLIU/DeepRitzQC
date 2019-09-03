import numpy as np
import torch


def rsubproblem(df_, lambda_, eta_):
    r_ = np.zeros(np.shape(df_))
    max_iter_ = 100
    for i in range(np.shape(df_)[0]):
        # df_i = torch.reshape(df_[i, :], 3, 3)
        # lambda_i = torch.reshape(lambda_[i, :], 3, 3)
        df_i = df_[i, :,:]
        lambda_i = lambda_[i, :, :]
        a_ = 2 * np.sum(np.diag(df_i) ** 2) / (3 * eta_)
        d_ = (np.linalg.det(df_i) ** 2) ** (1 / 3)
        y = np.zeros(3)
        index = 4
        b = df_i - lambda_i
        u, s, v = np.linalg.svd(b)
        x = s
        if np.linalg.det(b) < 0:
            index = np.argmin(x)

        diff = 1
        k = 0
        while np.abs(diff) > 1e-7 and k < max_iter_:
            d_last = d_
            if index == 4:
                y[0] = .5 * (x[0] + np.sqrt(x[0] ** 2 + 4 * a_ / d_))
                y[1] = .5 * (x[1] + np.sqrt(x[1] ** 2 + 4 * a_ / d_))
                y[2] = .5 * (x[2] + np.sqrt(x[2] ** 2 + 4 * a_ / d_))
            elif index == 0:
                y[0] = .5 * (x[0] - np.sqrt(x[0] ** 2 + 4 * a_ / d_))
                y[1] = .5 * (x[1] + np.sqrt(x[1] ** 2 + 4 * a_ / d_))
                y[2] = .5 * (x[2] + np.sqrt(x[2] ** 2 + 4 * a_ / d_))
            elif index == 1:
                y[0] = .5 * (x[0] + np.sqrt(x[0] ** 2 + 4 * a_ / d_))
                y[1] = .5 * (x[1] - np.sqrt(x[1] ** 2 + 4 * a_ / d_))
                y[2] = .5 * (x[2] + np.sqrt(x[2] ** 2 + 4 * a_ / d_))
            elif index == 2:
                y[0] = .5 * (x[0] + np.sqrt(x[0] ** 2 + 4 * a_ / d_))
                y[1] = .5 * (x[1] + np.sqrt(x[1] ** 2 + 4 * a_ / d_))
                y[2] = .5 * (x[2] - np.sqrt(x[2] ** 2 + 4 * a_ / d_))
            r = y[0] * y[1] * y[2]
            d_ = ((r/d_last)**2+4*d_last)/5

            diff = np.abs(d_last - d_)
            k += 1

        m = np.array([y[0], y[1], y[2]])
        fg = np.diag(m)
        r_[i, :, :] = np.matmul(np.matmul(u, fg), np.transpose(v))
    return r_

