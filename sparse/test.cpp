#include "../hnswlib/csr_matrix.h"

int main() {
  CSRMatrix a("/home/mr/big-ann-benchmarks/data/sparse/base_full.csr");
  a.save("/home/mr/hnswlib/sparse/base_full");
}