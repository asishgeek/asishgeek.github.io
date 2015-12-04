---
layout: post
title:  "Numpy tricks: Element-wise multplication of rows of two matrices" 
date:   2015-12-03
---

While working on a project I came across a situation where I had to do an element-wise 
multiplication of rows of two matrices to produce a 3rd-order tensor. A few minutes of
Googling did not give me an efficient solution. So here is a solution that is faster
than the naive approach of using python list comprehension.

## Problem
Let \\(\mathbf{A}\\) be a \\(m \times n\\) matrix and \\(\mathbf{B}\\) be a \\(p \times n\\) matrix.
We want a \\(m \times p \times n\\) tensor \\(\mathbf{C}\\) such that 
\\(\mathbf{C}\_{i,j} = \mathbf{A}\_i \circ \mathbf{B}\_j \\),
where \\(\mathbf{A}\_i\\) is the i-th row of A and \\(\mathbf{B}\_j\\) is the j-th row of B.

## Solution
Assume we have the following matrices:

    > A = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    > A
    array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])

    > B = arange(12).reshape((4,3))
    > B
    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]])

### Naive solution 
The following uses python list comprehension:

    > C = np.array([A[i,:] * B[j,:] for i in range(A.shape[0]) 
            for j in range(B.shape[0])]).reshape(
            (A.shape[0], B.shape[0], A.shape[1]))
    > C
    array([[[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11]],

           [[ 0,  2,  4],
            [ 6,  8, 10],
            [12, 14, 16],
            [18, 20, 22]],

           [[ 0,  3,  6],
            [ 9, 12, 15],
            [18, 21, 24],
            [27, 30, 33]]])       
    > C[2,3]
    array([27, 30, 33])

### Solution using einsum
The following uses [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) function
*einsum* of numpy:

    > C = einsum('ij,kj->ikj', A, B)
    > C 
    array([[[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11]],

           [[ 0,  2,  4],
            [ 6,  8, 10],
            [12, 14, 16],
            [18, 20, 22]],

           [[ 0,  3,  6],
            [ 9, 12, 15],
            [18, 21, 24],
            [27, 30, 33]]])       

## Performance
Not only is the einsum approach much shorter, it is also about 10 times faster than the
naive approach. Here are some quick benchmark results on my system running OS X.

    > %timeit -n 10000 array([A[i,:] * B[j,:] for i in range(A.shape[0]) for j in range(B.shape[0])]).reshape((A.shape[0], B.shape[0], A.shape[1]))
    10000 loops, best of 3: 28.9 µs per loop

    > %timeit -n 10000 einsum('ij,kj->ikj', A, B)
    10000 loops, best of 3: 2.84 µs per loop

In the first approach creating a new array and reshaping it are additional operations that the *einsum*
method doesn't have. So to get an idea about the time it takes to just multiply the rows using
python list comprehension I ran the following benchmark:

    > %timeit -n 10000 [A[i,:] * B[j,:] for i in range(A.shape[0]) for j in range(B.shape[0])]
    10000 loops, best of 3: 22.4 µs per loop

From the above it is clear that the bulk of the time is spent in looping, which is slow,
rather than creating a numpy array and reshaping it.

## Concluding remarks
The *einsum* function in numpy is a powerful construct that can be used
to represent complex matrix operations in a compact way and can result
in significant performance improvement over loops. It is
worth learning the Einstein sum notation, even for it's own sake &mdash; it 
is pretty cool after all.
