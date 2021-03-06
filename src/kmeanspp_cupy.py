#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
kmeanspp_cupy.py
=================================================================
Implements the kmeans++ algorithm for finding an initial codebook
for the subsequent kmeans iterations. Uses the CuPy library.
'''

import datetime
import time

import cupy as cp


def distances_squared(points_not_chosen_thus_far: cp.ndarray,
                      center: cp.ndarray):
    """
    Computes the squared norm of differences between all points
    in points_not_chosen_thus_far and the center point.

    :param points_not_chosen_thus_far: points not chosen thus far by the selection process \
    points are stored as columns
    :type points_not_chosen_thus_far: cp.ndarray
    :param center: newly chosen center point
    :type center: cp.ndarray
    """

    diffs = cp.transpose(points_not_chosen_thus_far) - center

    distances_squared = cp.sum(diffs * diffs, axis=1)

    return distances_squared


def kmeanspp(train_set: cp.ndarray, num_codevectors: int, verbose: bool = True):
    """
    Runs the kmeans++ algorithm. The computations precision is inherited from the train_set.

    :param train_set: training set, points as columns
    :type train_set: cp.ndarray
    :param num_codevectors: number of codevectors in the initial codebook
    :type num_codevectors: int
    :param verbose: print additional information while running
    :type verbose: bool
    """

    start = time.time()
    start_time = datetime.datetime.now()

    # choose first codevector
    remaining_points = cp.arange(train_set.shape[1]) # points to choose among

    i = cp.random.choice(remaining_points.shape[0], size=1)

    initial_codebook = cp.zeros((train_set.shape[0], num_codevectors), dtype = train_set.dtype)

    initial_codebook[:, 0] = train_set[:,remaining_points[i]].ravel()

    # choose remaining codevectors
    start_chunk = time.time()
    for n in range(1, num_codevectors):

        if verbose and time.time() - start_chunk > 5.0:

            secs_per_cv = (time.time()-start)/n

            eta = start_time + datetime.timedelta(seconds = secs_per_cv * num_codevectors)

            print(f"Number of selected codevectors yet {n}/{num_codevectors} ... ",
                  f"Elapsed time: {time.time()-start:.2f} [s] ... ETA {eta} ...")

            start_chunk = time.time()

        remaining_points = \
            cp.concatenate((remaining_points[:i],remaining_points[i+1:]), axis = 0)

        dist2 = distances_squared(train_set[:,remaining_points], initial_codebook[:, n - 1])

        p = dist2 / cp.sum(dist2)

        i = cp.random.choice(remaining_points.shape[0], size=1, p=p)

        initial_codebook[:, n] = train_set[:,remaining_points[i]].ravel()

    end = time.time()

    if verbose:
        print(f"Elapsed time: {end-start:.2f} [s]")

    return initial_codebook


if __name__ == "__main__":

    ts = cp.random.randn(10, 10000000).astype(cp.float32)

    kmeanspp(ts, 1000)
