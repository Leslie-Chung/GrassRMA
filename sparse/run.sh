#!/bin/bash
# g++ -g -o ./build_index ./build_index.cpp -lpthread
g++ -o ./search_index ./search_index.cpp -lpthread

data_set=("base_full")

for data in "${data_set[@]}"; do
  # ./build_index -e 500 -m 16 -t 64 \
  #   -d /home/mr/big-ann-benchmarks/data/sparse/base_full.csr \
  #   -i /home/mr/hnswlib/sparse/indices/reorder1/base_full

  ./app/search_index_combined -e 73 -i /home/mr/hnswlib/sparse/indices/noreorder/base_full \
    -q /home/mr/big-ann-benchmarks/data/sparse/queries.dev.csr \
    -g /home/mr/big-ann-benchmarks/data/sparse/base_full.dev.gt -t 8 \
    -r /home/mr/hnswlib/sparse/results/base_full/base_full.log

  ./search_index -e 75 -i /home/mr/hnswlib/sparse/indices/reorder1/base_full \
    -q /home/mr/big-ann-benchmarks/data/sparse/queries.dev.csr \
    -g /home/mr/big-ann-benchmarks/data/sparse/base_full.dev.gt -t 8 \
    -r /home/mr/hnswlib/sparse/results/base_full/base_full.log

  ./search_index -e 75 -i /home/mr/hnswlib/sparse/indices/reorder/base_full \
    -q /home/mr/big-ann-benchmarks/data/sparse/queries.dev.csr \
    -g /home/mr/big-ann-benchmarks/data/sparse/base_full.dev.gt -t 8 \
    -r /home/mr/hnswlib/sparse/results/base_full/base_full.log
done


