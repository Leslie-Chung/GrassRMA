#pragma once
#include "hnswlib.h"

namespace sparse_hnswlib
{

    static float
    InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr, const void *other_ptr)
    {
        CSRMatrix *csr_matrix = reinterpret_cast<CSRMatrix *>(const_cast<void *>(qty_ptr));
        
        const uint32_t q_idx = *((uint32_t *) pVect1);
        const uint32_t p_idx = *((uint32_t *) pVect2);

        const uint32_t p_start = csr_matrix->indptr[p_idx];
        const uint32_t p_end = csr_matrix->indptr[p_idx + 1];
        IndiceDataPair *p_indices = csr_matrix->indices_data + p_start;

        uint32_t q_end, q_start = 0;
        IndiceDataPair *q_indices;
        if (other_ptr == nullptr) {
            q_start = csr_matrix->indptr[q_idx];
            q_end = csr_matrix->indptr[q_idx + 1];
            q_indices = csr_matrix->indices_data + q_start;
        } else {
            CSRMatrix *csr_matrix_query = reinterpret_cast<CSRMatrix *>(const_cast<void *>(other_ptr));
            q_start = csr_matrix_query->indptr[q_idx];
            q_end = csr_matrix_query->indptr[q_idx + 1];
            q_indices = csr_matrix_query->indices_data + q_start;
        }

        uint32_t p_num = p_end - p_start;
        uint32_t q_num = q_end - q_start;
        uint32_t i = 0, j = 0; 

        // int32_t q_min = q_indices[0], q_max = q_indices[q_num - 1];
        // int32_t p_min = p_indices[0], p_max = p_indices[p_num - 1];
        // if (q_min > p_max || p_min > q_max) {
        //     return 0;
        // }
        // IndiceDataPair q_min = q_indices[0], q_max = q_indices[q_num - 1];
        // IndiceDataPair p_min = p_indices[0], p_max = p_indices[p_num - 1];
        // if (q_min > p_max || p_min > q_max) {
        //     return 0;
        // }

        // if(q_min < p_min) { // 让q的列靠近p的列
        //     i = std::lower_bound(q_indices, q_indices + q_num, p_min) - q_indices;
        // } else if(p_min < q_min) { // 让p的列靠近q的列
        //     j = std::lower_bound(p_indices, p_indices + p_num, q_min) - p_indices;
        // }

        // if(q_max > p_max) {
        //     q_num = std::lower_bound(q_indices, q_indices + q_num, p_max) - q_indices + 1;
        // } else if(p_max > q_max) {
        //     p_num = std::lower_bound(p_indices, p_indices + p_num, q_max) - p_indices + 1;
        // }

        float res = 0;
        
        // while (i < q_num && j < p_num)
        // {
        //     if (q_indices[i] < p_indices[j]) ++i; 
        //     else if (q_indices[i] > p_indices[j]) ++j;
        //     else
        //     {
        //         res += q_data[i] * p_data[j];
        //         ++i;
        //         ++j;
        //     }
        // }
        // _mm_prefetch((char *) (q_indices), _MM_HINT_T2);
        // _mm_prefetch((char *) (q_indices + 8), _MM_HINT_T2);
        // _mm_prefetch((char *) (q_indices + 16), _MM_HINT_T2);
        // _mm_prefetch((char *) (p_indices), _MM_HINT_T2);
        // _mm_prefetch((char *) (p_indices + 8), _MM_HINT_T2);
        // _mm_prefetch((char *) (p_indices + 16), _MM_HINT_T2);

        int32_t q_col = q_indices->indice;
        int32_t p_col = p_indices->indice;
        IndiceDataPair* q_indices_end = q_indices + q_num;
        IndiceDataPair* p_indices_end = p_indices + p_num;

        while (q_indices < q_indices_end && p_indices < p_indices_end)
        {
            if (q_col < p_col) {
                q_col = (++q_indices)->indice; 
            }
            else if (q_col > p_col) {
                p_col = (++p_indices)->indice; 
            }
            else
            {
                res += q_indices->data * p_indices->data;
                q_col = (++q_indices)->indice; 
                p_col = (++p_indices)->indice;  
                // res += *(q_data++) * *(p_data++);
                // q_col = *(++q_indices); 
                // p_col = *(++p_indices);  
            }
        }
        return res;
    }

    static float
    InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr, const void *other_ptr)
    {
        return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr, other_ptr);
    }



    class InnerProductSpace : public SpaceInterface<float>
    {
        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;

    public:
        InnerProductSpace(size_t dim)
        {
            fstdistfunc_ = InnerProductDistance;
            dim_ = dim;
            data_size_ = dim * sizeof(float);
        }

        size_t get_data_size()
        {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func()
        {
            return fstdistfunc_;
        }

        void *get_dist_func_param()
        {
            return &dim_;
        }

        ~InnerProductSpace() {}
    };

} // namespace sparse_hnswlib
