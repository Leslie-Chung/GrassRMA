g++ -o ./build_index ./build_index.cpp -lpthread
# g++ -o ./search_index_combined ./search_index.cpp -lpthread

data_set=("base_full")

for data in "${data_set[@]}"; do
  ./build_index -e 20 -m 2 -t 2 \
    -d /home/ubuntu/big-ann-benchmarks/data/sparse/${data}.csr \
    -i /home/ubuntu/SparseHNSW/sparse/${data}

  # ./search_index_combined -e 90 -i /home/ubuntu/SparseHNSW/sparse/${data} \
  #   -q /home/ubuntu/big-ann-benchmarks/data/sparse/queries.dev.csr \
  #   -g /home/ubuntu/big-ann-benchmarks/data/sparse/${data}.dev.gt -t 1 \
  #   -r /home/ubuntu/SparseHNSW/sparse/${data}.log

done


