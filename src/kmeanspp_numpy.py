#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
kmeanspp_numpy.py
=================================================================
Implements the kmeans++ algorithm for finding an initial codebook
for the subsequent kmeans iterations. Uses the NumPy library.
'''

import datetime
import time

import numpy as cp


def distances_squared(points_not_choosed_thus_far: cp.ndarray,
                      center: cp.ndarray):
    """distances_squared.
    Computes the squared norm of differences between all points
    in points_not_choosed_thus_far and the center point.

    :param points_not_choosed_thus_far: points not choosen thus far by the selection process
    points are stored as columns
    :type points_not_choosed_thus_far: numpy.ndarray
    :param center: newly choosen center point
    :type center: numpy.ndarray
    """

    diffs = cp.transpose(points_not_choosed_thus_far) - center

    distances_squared = cp.sum(diffs * diffs, axis=1)

    return distances_squared


def kmeanspp(train_set: cp.ndarray, num_codevectors: int, verbose: bool = True):
    """kmeanspp.
    Runs the kmeans++ algorithm. The computations precision is inherited from the train_set.

    :param train_set: training set, points as columns
    :type train_set: numpy.ndarray
    :param num_codevectors: number of codevectors in the in the initial codebook
    :type num_codevectors: int
    :param verbose: print additional information while running
    :type verbose: bool
    """

    start = time.time()
    start_time = datetime.datetime.now()

    # choose first codevector
    remaining_points = train_set  # points to choose among

    i = cp.random.choice(remaining_points.shape[1], size=1)

    initial_codebook = cp.zeros((train_set.shape[0], num_codevectors), dtype = train_set.dtype)

    initial_codebook[:, 0] = remaining_points[:, i].ravel()

    # choose remaining codevectors
    start_chunk = time.time()
    for n in range(1, num_codevectors):

        if verbose and time.time() - start_chunk > 5.0:

            secs_per_cv = (time.time()-start)/n

            eta = start_time + datetime.timedelta(seconds = secs_per_cv * num_codevectors)

            print(f"Number of selected codevectors yet {n}/{num_codevectors} ... ",
                  f"Elapsed time: {time.time()-start:.2f} [s] ... ETA {eta} ...")

            start_chunk = time.time()

        remaining_points = cp.delete(remaining_points,[i],axis=1)

        dist2 = distances_squared(remaining_points, initial_codebook[:, n - 1])

        p = dist2 / cp.sum(dist2)

        i = cp.random.choice(remaining_points.shape[1], size=1, p=p)

        initial_codebook[:, n] = remaining_points[:, i].ravel()

    end = time.time()

    if verbose:
        print(f"Elapsed time: {end-start:.2f} [s]")

    return initial_codebook


if __name__ == "__main__":

    ts = cp.random.randn(10, 10000000).astype(cp.float32)

    kmeanspp(ts, 1000)
