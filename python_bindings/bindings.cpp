#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "hnswlib.h"
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <assert.h>

namespace py = pybind11;
using namespace pybind11::literals;  // needed to bring in _a literal

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


inline void assert_true(bool expr, const std::string & msg) {
    if (expr == false) throw std::runtime_error("Unpickle Error: " + msg);
    return;
}


class CustomFilterFunctor: public sparse_hnswlib::BaseFilterFunctor {
    std::function<bool(sparse_hnswlib::labeltype)> filter;

 public:
    explicit CustomFilterFunctor(const std::function<bool(sparse_hnswlib::labeltype)>& f) {
        filter = f;
    }

    bool operator()(sparse_hnswlib::labeltype id) {
        return filter(id);
    }
};


inline std::vector<size_t> get_input_ids_and_check_shapes(const py::object& ids_, size_t feature_rows) {
    std::vector<size_t> ids;
    if (!ids_.is_none()) {
        py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
        auto ids_numpy = items.request();
        // check shapes
        // extract data
        if (ids_numpy.ndim == 1) {
            std::vector<size_t> ids1(ids_numpy.shape[0]);
            for (size_t i = 0; i < ids1.size(); i++) {
                ids1[i] = items.data()[i];
            }
            ids.swap(ids1);
        } else if (ids_numpy.ndim == 0) {
            ids.push_back(*items.data());
        }
    }

    return ids;
}


template<typename dist_t, typename data_t = float>
class Index {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    size_t seed;
    size_t default_ef;

    bool index_inited;
    bool ep_added;
    int num_threads_default;
    sparse_hnswlib::labeltype cur_l;
    sparse_hnswlib::HierarchicalNSW<dist_t>* appr_alg;
    sparse_hnswlib::SpaceInterface<float>* l2space;

    Index(const std::string &space_name, const int dim) : space_name(space_name), dim(dim) {
        if (space_name == "l2") {
            l2space = new sparse_hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            l2space = new sparse_hnswlib::InnerProductSpace(dim);
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        appr_alg = NULL;
        ep_added = true;
        index_inited = false;
        num_threads_default = std::thread::hardware_concurrency();

        default_ef = 10;
    }


    ~Index() {
        delete l2space;
        if (appr_alg)
            delete appr_alg;
    }


    void init_new_index(
        size_t maxElements,
        const std::string &csr_path,
        size_t M,
        size_t efConstruction,
        size_t random_seed,
        bool allow_replace_deleted) {
        if (appr_alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new sparse_hnswlib::HierarchicalNSW<dist_t>(l2space, new CSRMatrix(csr_path), maxElements, M, efConstruction, random_seed, allow_replace_deleted);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        seed = random_seed;
    }


    void set_ef(size_t ef) {
      default_ef = ef;
      if (appr_alg)
          appr_alg->ef_ = ef;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
    }


    void saveIndex(const std::string &path_to_index) {
        appr_alg->gorder(5, path_to_index);
        appr_alg->saveIndex(path_to_index);
    }


    void loadIndex(const std::string &path_to_index, size_t max_elements, bool allow_replace_deleted) {
      if (appr_alg) {
          std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
          delete appr_alg;
      }
      appr_alg = new sparse_hnswlib::HierarchicalNSW<dist_t>(l2space, path_to_index, false, max_elements, allow_replace_deleted);
      cur_l = appr_alg->cur_element_count;
      index_inited = true;
    }

    void addItems(py::object ids_ = py::none(), int num_threads = -1, bool replace_deleted = false) {
        if (num_threads <= 0)
            num_threads = num_threads_default;

        size_t rows = appr_alg->csr_matrix_->nrow;

        // avoid using threads when the number of additions is small:
        if (rows <= num_threads * 4) {
            num_threads = 1;
        }

        std::vector<size_t> ids(rows);
        for (size_t i = 0; i < rows; i++) {
            ids[i] = i;
        }
        {
            int start = 0;
            if (!ep_added) {
                size_t id = ids.size() ? ids.at(0) : (cur_l);
                appr_alg->addPoint((size_t)id, (size_t)id, replace_deleted);
                start = 1;
                ep_added = true;
            }

            py::gil_scoped_release l;
            ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                appr_alg->addPoint((size_t)id, (size_t)id, replace_deleted);
            });
            cur_l += rows;
        }
    }


    std::vector<sparse_hnswlib::labeltype> getIdsList() {
        std::vector<sparse_hnswlib::labeltype> ids;

        for (auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }

    py::object knnQuery_return_numpy(
        const py::array_t<int64_t>& indptr,
        const py::array_t<int32_t>& indices,
        const py::array_t<float>& data,
        size_t k = 1,
        int num_threads = -1,
        const std::function<bool(sparse_hnswlib::labeltype)>& filter = nullptr) {
        sparse_hnswlib::labeltype* data_numpy_l;
        dist_t* data_numpy_d;
        size_t rows;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            auto r_indptr = indptr.unchecked<1>();
            auto r_indices = indices.unchecked<1>();
            auto r_data = data.unchecked<1>();

            int64_t nrow = r_indptr.shape(0) - 1;
            rows = nrow;
            int64_t nnz = r_indices.shape(0);
                    // avoid using threads when the number of searches is small:
            if (rows <= num_threads * 4) {
                num_threads = 1;
            }
            CSRMatrix query_csr_matrix(nrow, -1, nnz, r_indptr.data(0), r_indices.data(0), r_data.data(0));
            data_numpy_l = new sparse_hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];

            CustomFilterFunctor* p_idFilter = nullptr;

            ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                std::priority_queue<std::pair<dist_t, sparse_hnswlib::labeltype >> result = appr_alg->searchKnn(
                    row, k, &query_csr_matrix, p_idFilter);
                if (result.size() != k)
                    throw std::runtime_error(
                        "Cannot return the results in a contigious 2D array. Probably ef or M is too small");
                for (int i = k - 1; i >= 0; i--) {
                    auto& result_tuple = result.top();
                    data_numpy_d[row * k + i] = result_tuple.first;
                    data_numpy_l[row * k + i] = result_tuple.second;
                    result.pop();
                }
            });
        }
        py::capsule free_when_done_l(data_numpy_l, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) {
            delete[] f;
            });
        return py::make_tuple(
            py::array_t<sparse_hnswlib::labeltype>(
                { rows, k },  // shape
                { k * sizeof(sparse_hnswlib::labeltype),
                  sizeof(sparse_hnswlib::labeltype) },  // C-style contiguous strides for each index
                data_numpy_l,  // the data pointer
                free_when_done_l),
            py::array_t<dist_t>(
                { rows, k },  // shape
                { k * sizeof(dist_t), sizeof(dist_t) },  // C-style contiguous strides for each index
                data_numpy_d,  // the data pointer
                free_when_done_d));
    }

    size_t getMaxElements() const {
        return appr_alg->max_elements_;
    }

    size_t getCurrentCount() const {
        return appr_alg->cur_element_count;
    }
};


PYBIND11_PLUGIN(sparse_hnswlib) {
        py::module m("sparse_hnswlib");

        py::class_<Index<float>>(m, "Index")
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index",
            &Index<float>::init_new_index,
            py::arg("max_elements"),
            py::arg("csr_path"),
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("random_seed") = 100,
            py::arg("allow_replace_deleted") = false)
        .def("knn_query",
            &Index<float>::knnQuery_return_numpy,
            py::arg("indptr"),
            py::arg("indices"),
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1,
            py::arg("filter") = py::none())
        .def("add_items",
            &Index<float>::addItems,
            py::arg("ids") = py::none(),
            py::arg("num_threads") = -1,
            py::arg("replace_deleted") = false)
        .def("get_ids_list", &Index<float>::getIdsList)
        .def("set_ef", &Index<float>::set_ef, py::arg("ef"))
        .def("set_num_threads", &Index<float>::set_num_threads, py::arg("num_threads"))
        .def("save_index", &Index<float>::saveIndex, py::arg("path_to_index"))
        .def("load_index",
            &Index<float>::loadIndex,
            py::arg("path_to_index"),
            py::arg("max_elements") = 0,
            py::arg("allow_replace_deleted") = false)
        .def("get_max_elements", &Index<float>::getMaxElements)
        .def("get_current_count", &Index<float>::getCurrentCount)
        .def_readonly("space", &Index<float>::space_name)
        .def_readonly("dim", &Index<float>::dim)
        .def_readwrite("num_threads", &Index<float>::num_threads_default)
        .def_property("ef",
          [](const Index<float> & index) {
            return index.index_inited ? index.appr_alg->ef_ : index.default_ef;
          },
          [](Index<float> & index, const size_t ef_) {
            index.default_ef = ef_;
            if (index.appr_alg)
              index.appr_alg->ef_ = ef_;
        })
        .def_property_readonly("max_elements", [](const Index<float> & index) {
            return index.index_inited ? index.appr_alg->max_elements_ : 0;
        })
        .def_property_readonly("element_count", [](const Index<float> & index) {
            return index.index_inited ? (size_t)index.appr_alg->cur_element_count : 0;
        })
        .def_property_readonly("ef_construction", [](const Index<float> & index) {
          return index.index_inited ? index.appr_alg->ef_construction_ : 0;
        })
        .def_property_readonly("M",  [](const Index<float> & index) {
          return index.index_inited ? index.appr_alg->M_ : 0;
        })


        .def("__repr__", [](const Index<float> &a) {
            return "<sparse_hnswlib.Index(space='" + a.space_name + "', dim="+std::to_string(a.dim)+")>";
        });

        return m.ptr();
}
