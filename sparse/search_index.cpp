#include "../hnswlib/hnswlib.h"
#include <chrono>
#include <getopt.h>
#include <thread>
#include <algorithm>

uint32_t n, d;
static void get_gt(const std::string gt_path, uint32_t *&I)
{
    std::ifstream infile(gt_path, std::ios::binary);

    if (infile.fail())
    {
        std::cerr << std::string("Failed to open file ") + gt_path;
        exit(1);
    }
    infile.read((char *)&n, sizeof(uint32_t));
    infile.read((char *)&d, sizeof(uint32_t));
    I = new uint32_t[n * d];
    infile.read((char *)I, n * d * sizeof(uint32_t));
    infile.close();
}

int recall(std::priority_queue<std::pair<float, sparse_hnswlib::labeltype>> &result, uint32_t *I)
{
    int ret = 0;
    uint32_t *end = I + d;
    while (!result.empty())
    {
        uint32_t p = result.top().second;
        if (std::find(I, end, p) != end)
        {
            ++ret;
        }
        result.pop();
    }
    return ret;
}

int main(int argc, char *argv[])
{
    const struct option longopts[] = {
        {"ef", required_argument, 0, 'e'},

        // Indexing Path
        {"index_path", required_argument, 0, 'i'},
        {"query_path", required_argument, 0, 'q'},
        {"groundtruth_path", required_argument, 0, 'g'},
        {"threads num", required_argument, 0, 't'},
        {"result_path", required_argument, 0, 'r'},
    };

    int ind;
    int iarg = 0;
    size_t ef = 10;

    char index_path[256] = "";
    char query_path[256] = "";
    char groundtruth_path[256] = "";
    char result_path[256] = "";
    int nthreads = 16;

    while (iarg != -1)
    {
        iarg = getopt_long(argc, argv, "e:i:q:g:t:r:", longopts, &ind);
        switch (iarg)
        {
        case 'e':
            if (optarg)
                ef = atoi(optarg);
            break;
        case 'g':
            if (optarg)
                strcpy(groundtruth_path, optarg);
            break;
        case 'q':
            if (optarg)
                strcpy(query_path, optarg);
            break;
        case 'i':
            if (optarg)
                strcpy(index_path, optarg);
            break;
        case 'r':
            if (optarg)
                strcpy(result_path, optarg);
            break;
        case 't':
            if (optarg)
                nthreads = atoi(optarg);
            break;
        }
    }

    CSRMatrix query_csr_matrix(query_path);

    int dim = 16; // Dimension of the elements

    // Initing index
    sparse_hnswlib::InnerProductSpace space(dim);
    // std::string hnsw_path = "/home/mr/hnswlib/sparse/indices/sparse-small/base_small";
    sparse_hnswlib::HierarchicalNSW<float> *alg_hnsw = new sparse_hnswlib::HierarchicalNSW<float>(&space, index_path);
    uint32_t *I = nullptr;
    get_gt(groundtruth_path, I);
    alg_hnsw->setEf(ef);
    freopen(result_path, "a", stdout);

    float correct = 0;
    double total_pqs = 0;

    std::vector<std::thread> threads;
    std::vector<double> qps;
    std::vector<float> corrects;
    auto process_data = [&](uint32_t start, uint32_t end, uint32_t thread_i)
    {
        double total_duration = 0.0;
        for (uint32_t i = start; i < end; i++)
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            std::priority_queue<std::pair<float, sparse_hnswlib::labeltype>> result = alg_hnsw->searchKnn(i, d, &query_csr_matrix);
            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            total_duration += duration.count();
            corrects[thread_i] += recall(result, I + i * d);
        }
        qps[thread_i] = (end - start) / total_duration * 1e6;
    };

    uint32_t chunk_size = n / nthreads;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < nthreads; i++)
    {
        uint32_t start = i * chunk_size;
        uint32_t end = (i == nthreads - 1) ? n : (i + 1) * chunk_size;
        threads.emplace_back(process_data, start, end, i);
        qps.emplace_back(0);
        corrects.emplace_back(0);
    }

    for (auto &thread : threads)
    {
        thread.join();
    }
// for (uint32_t i = 0; i < n; i++)
//         {
//             std::priority_queue<std::pair<float, sparse_hnswlib::labeltype>> result = alg_hnsw->searchKnn(i, d, &query_csr_matrix);
//         }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    float QPS = 1.0f * n / duration.count() * 1e6;
    for (uint32_t i = 0; i < nthreads; i++)
    {
        total_pqs += qps[i];
        correct += corrects[i];
    }

    float recall = (float)correct / (n * d) * 100;
    std::cout << "Recall: " << recall << "\n";
    std::cout << "QPS: " << total_pqs << "\n";
    std::cout << "QPS_new: " << QPS << "\n";
    delete alg_hnsw;
    delete I;
    return 0;
}