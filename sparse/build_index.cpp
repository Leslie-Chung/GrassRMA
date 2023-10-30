#include "../hnswlib/hnswlib.h"
#include <thread>
#include <getopt.h>
int main(int argc, char *argv[]) {
    const struct option longopts[] = {
        // Query Parameter
        {"efConstruction", required_argument, 0, 'e'},
        {"M", required_argument, 0, 'm'},
        {"threads num", no_argument, 0, 't'},
        // Indexing Path
        {"dataset", required_argument, 0, 'd'},
        {"index_path", required_argument, 0, 'i'},
    };

    char index_path[256] = "";
    char data_path[256] = "";

    int nthreads = 32;
    size_t ef_construction = 200;
    size_t M = 16;

    int ind;
    int iarg = 0;

    while (iarg != -1)
    {
        iarg = getopt_long(argc, argv, "e:d:i:m:t:", longopts, &ind);
        switch (iarg)
        {
        case 'e':
            if (optarg) ef_construction = atoi(optarg);
            break;
        case 'm':
            if (optarg) M = atoi(optarg);
            break;
        case 'd':
            if (optarg) strcpy(data_path, optarg);
            break;
        case 'i':
            if (optarg) strcpy(index_path, optarg);
            break;
        case 't':
            if (optarg) nthreads = atoi(optarg);
            break;
        }
    }



    CSRMatrix *csr_matrix = new CSRMatrix(data_path);
    
    int dim = 16;               // Dimension of the elements
    uint32_t max_elements = csr_matrix->nrow;   // Maximum number of elements, should be known beforehand

    // Initing index
    sparse_hnswlib::InnerProductSpace space(dim);
    sparse_hnswlib::HierarchicalNSW<float>* alg_hnsw = new sparse_hnswlib::HierarchicalNSW<float>(&space, csr_matrix, max_elements, M, ef_construction);
    // Add data to index

    auto process_data = [&](uint32_t start, uint32_t end)
    {
        for (uint32_t i = start; i < end; i++)
        {
            alg_hnsw->addPoint(i, i);
        }
    };

    std::vector<std::thread> threads;
    uint32_t chunk_size = max_elements / nthreads;
    for (uint32_t i = 0; i < nthreads; i++)
    {
        uint32_t start = i * chunk_size;
        uint32_t end = (i == nthreads - 1) ? max_elements : (i + 1) * chunk_size;
        threads.emplace_back(process_data, start, end);
    }

    // 等待所有线程完成
    for (auto &thread : threads)
    {
        thread.join();
    }

    alg_hnsw->saveIndex(index_path);

    delete alg_hnsw;
    return 0;
}