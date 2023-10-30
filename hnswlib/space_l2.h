#pragma once
#include "hnswlib.h"

namespace sparse_hnswlib {

static float
L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const void *other_ptr = nullptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}



class L2Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2Space(size_t dim) {
        fstdistfunc_ = L2Sqr;
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2Space() {}
};


static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr, const void* __restrict other_ptr=nullptr) {
    size_t qty = *((size_t*)qty_ptr);
    int res = 0;
    unsigned char* a = (unsigned char*)pVect1;
    unsigned char* b = (unsigned char*)pVect2;

    for (size_t i = 0; i < qty; i++) {
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
    }
    return (res);
}

class L2SpaceI : public SpaceInterface<int> {
    DISTFUNC<int> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2SpaceI(size_t dim) {
        fstdistfunc_ = L2SqrI;
        
        dim_ = dim;
        data_size_ = dim * sizeof(unsigned char);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<int> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2SpaceI() {}
};
}  // namespace sparse_hnswlib
