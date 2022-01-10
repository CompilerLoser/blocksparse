import numpy as np
import scipy.sparse as sparse


def xn_lut( ys, xs, blocks, ctx_blks):

    # build list of y's connected to each x and map to block id
    py_lut = [list() for y in range(ctx_blks)]
    for b in range(blocks):
        py_lut[ ys[b] ].append(( b, xs[b] ))

    # build header into variable lengh lookup tables (luts)
    # the header contains the offset and size of the lut for that output block
    max_lut = 0
    offset  = ctx_blks
    np_lut  = np.empty((offset + blocks, 2), dtype=np.int32)

    for i, lut in enumerate(py_lut):
        np_lut[i] = offset,  len(lut)
        max_lut = max(max_lut, len(lut))
        for entry in lut:
            np_lut[offset] = entry
            offset += 1

    return np_lut, py_lut, max_lut



a = [[1,1,0],[1,1,1],[0,1,1]]
b= np.array(a)
csr = sparse.csr_matrix(b)
ys, xs, bs = sparse.find(csr)
nt_list = sorted(zip(ys, xs))
print(nt_list)

ys = [b[0] for b in nt_list]
xs = [b[1] for b in nt_list]
nt_list = np.array(nt_list, dtype=np.int32)
print(bs)
print(ys)
print(xs)
np_lut, py_lut, max_lut = xn_lut(ys, xs, 7, 3)
print(np_lut)
print(py_lut)
print(max_lut)
print("*************")
for i, lut in enumerate(py_lut):
    print(i)
    print(lut)
