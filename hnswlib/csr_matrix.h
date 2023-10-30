
#pragma once
#ifndef MATRIX_HPP_
#define MATRIX_HPP_
#include <cstddef>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <cstring>
#include <assert.h>

struct IndiceDataPair
{
  int32_t indice;
  float data;
  bool operator> (const IndiceDataPair& s2)//  排名升序
  {	
      return this->indice > s2.indice;
  }
  bool operator< (const IndiceDataPair& s2)//  排名升序
  {	
      return this->indice < s2.indice;
  }

  bool operator== (const IndiceDataPair& s2)//  排名升序
  {	
      return this->indice == s2.indice;
  }
};

class CSRMatrix
{
private:
public:
  int64_t nrow{0};
  int64_t ncol{0};
  int64_t nnz{0}; // 稀疏矩阵中的非零元素的总数
  int64_t *indptr{nullptr};

  IndiceDataPair *indices_data{nullptr};

  CSRMatrix(const std::string data_file_path, bool ori = true)
  {

    std::ifstream infile(data_file_path, std::ios::binary);

    if (infile.fail())
    {
      std::cerr << std::string("Failed to open file ") + data_file_path;
      exit(1);
    }

    infile.read((char *)&nrow, sizeof(int64_t));
    infile.read((char *)&ncol, sizeof(int64_t));
    infile.read((char *)&nnz, sizeof(int64_t));

    indptr = new int64_t[nrow + 1];
    infile.read((char *)indptr, (nrow + 1) * sizeof(int64_t));
    indices_data = new IndiceDataPair[nnz];

    if (ori)
    {
      int32_t *indices = new int32_t[nnz]; // 对应data在原矩阵的列
      infile.read((char *)indices, nnz * sizeof(int32_t));

      for (uint32_t i = 0; i < nnz; ++i)
      {
        indices_data[i].indice = indices[i];
      }
      delete[] indices;

      float *data = new float[nnz];
      infile.read((char *)data, nnz * sizeof(float));
      infile.close();

      for (uint32_t i = 0; i < nnz; ++i)
      {
        indices_data[i].data = data[i];
      }

      delete[] data;
    }
    else
    {
      infile.read((char *)indices_data, nnz * sizeof(IndiceDataPair));
    }
  }

  CSRMatrix(int64_t nrow_, int64_t ncol_, int64_t nnz_, const int64_t *indptr_, const int32_t *indices_, const float *data_)
  {
    nrow = nrow_;
    ncol = ncol_;
    nnz = nnz_;
    indptr = new int64_t[nrow + 1];

    indices_data = new IndiceDataPair[nnz];

    for (uint32_t i = 0; i < nnz; ++i)
    {
      indices_data[i].indice = indices_[i];
      indices_data[i].data = data_[i];
    }

    memcpy(indptr, indptr_, sizeof(uint64_t) * (nrow + 1));
  }

  // std::pair<uint32_t, uint32_t> get_row(uint32_t i)
  // {
  //   uint32_t start = indptr[i];
  //   uint32_t end = indptr[i + 1];
  //   return {start, end};
  // }

  ~CSRMatrix()
  {
    delete[] indptr, indices_data;
  }

  void save(const std::string location)
  {
    std::ofstream outfile(location, std::ios::binary);
    outfile.write((char *)&nrow, sizeof(int64_t));
    outfile.write((char *)&ncol, sizeof(int64_t));
    outfile.write((char *)&nnz, sizeof(int64_t));

    outfile.write((char *)indptr, (nrow + 1) * sizeof(int64_t));

    outfile.write((char *)indices_data, nnz * sizeof(IndiceDataPair));
    outfile.close();
  }
};

#endif