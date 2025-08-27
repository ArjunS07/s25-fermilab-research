# Implementation of Algorithm 3 from https://arxiv.org/abs/2312.07168


import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R

def align_point_clouds_till_converge(x_0_orig, x1, max_iter=2_500):
    x_0 = x_0_orig.clone()
    i = 0
    dist = np.linalg.norm(x_0 - x1, axis=1).sum()
    dist_delta = np.inf
    while i < max_iter and dist_delta > 1e-8:
        cost = cdist(x_0, x1, metric='euclidean')
        # Better for stability
        cost = cost * (1_000. / cost.max())
        _, col_ind = linear_sum_assignment(cost)

        x_0 = x_0[col_ind]
        # Align cartesian 3-momenta
        rot, _, _ = R.align_vectors(x1[:, 1:4], x_0[:, 1:4], return_sensitivity=True)
        x_0[:, 1:4] = x_0[:, 1:4] @ rot.as_matrix().T

        dist_new = np.linalg.norm(x_0 - x1, axis=1).sum()
        dist_delta = np.abs(dist_new - dist)
        dist = dist_new
        i += 1
    print(f"{i=} {dist=} {dist_delta=}")