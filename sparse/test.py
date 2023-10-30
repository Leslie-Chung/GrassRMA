import sparse_hnswlib
import numpy as np
import time

n = 8841823  # 100000 1000000 8841823
p = sparse_hnswlib.Index(space="ip", dim=16)
# p.init_index(
#     max_elements=n,
#     csr_path="/home/ubuntu/big-ann-benchmarks/data/sparse/base_full.csr",
#     ef_construction=1000,
#     M=16,
# )

# print("start")
# p.add_items(num_threads=8)
# p.save_index("/home/ubuntu/SparseHNSW/sparse/indices/base_full")
# print("end")

p.load_index("/home/ubuntu/SparseHNSW/sparse/indices/base_full", n)
p.set_ef(48)
def knn_result_read(fname):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    f = open(fname, "rb")
    f.seek(4+4)
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return I, D
  
def mmap_sparse_matrix_fields(fname):
    """ mmap the fields of a CSR matrix without instanciating it """
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
    ofs = sizes.nbytes
    indptr = np.memmap(fname, dtype='int64', mode='r', offset=ofs, shape=nrow + 1)
    ofs += indptr.nbytes
    indices = np.memmap(fname, dtype='int32', mode='r', offset=ofs, shape=nnz)
    ofs += indices.nbytes
    data = np.memmap(fname, dtype='float32', mode='r', offset=ofs, shape=nnz)
    return data, indices, indptr, ncol
  
data, indices, indptr, _ = mmap_sparse_matrix_fields("/home/ubuntu/big-ann-benchmarks/data/sparse/queries.dev.csr")
  
I, _ = knn_result_read("/home/ubuntu/big-ann-benchmarks/data/sparse/base_full.dev.gt")
start = time.time()
res, distances = p.knn_query(indptr, indices, data, k=10, num_threads=8)
end = time.time()
elapsed = end - start
intersection_sizes = np.array([np.intersect1d(row1, row2).size for row1, row2 in zip(I, res)])
print(f'Elapsed time: {elapsed}; {round(I.shape[0] /elapsed, 2)} QPS')
print(f'Recall: {np.sum(intersection_sizes) / (I.shape[0] * I.shape[1]) * 100}')