/**
 *  Example code for IO, read binary data vectors and save KNNs to path.
 *
 */

#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
////jf
#include <malloc.h>


#include "assert.h"
using std::cout;
using std::endl;
using std::string;
using std::vector;



#ifdef __GNUC__
#ifdef __AVX__
#define KGRAPH_MATRIX_ALIGN 32
#else
#ifdef __SSE2__
#define KGRAPH_MATRIX_ALIGN 16
#else
#define KGRAPH_MATRIX_ALIGN 4
#endif
#endif
#endif



/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knn a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void SaveKNN(const std::vector<std::vector<uint32_t>> &knns,
              const std::string &path = "output.bin") {
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
  const int K = 100;
  const uint32_t N = knns.size();
  assert(knns.front().size() == K);
  for (unsigned i = 0; i < N; ++i) {
    auto const &knn = knns[i];
    if (knn.size()) ////may be cause some bug here.
      ofs.write(reinterpret_cast<char const *>(&knn[0]), K * sizeof(uint32_t));
  }
  ofs.close();
}

void SaveBin(const std::string &file_path,
             const std::vector<std::vector<float>> &data) {
    std::cout << "Saving Data: " << file_path << std::endl;
    std::ofstream ofs;
    ofs.open(file_path, std::ios::binary);
    assert(ofs.is_open());
    uint32_t N = static_cast<uint32_t>(data.size()); // Number of points
    ofs.write(reinterpret_cast<const char*>(&N), sizeof(uint32_t));
    for (const auto& row : data) {
        ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }
    ofs.close();
    std::cout << "Finish Saving Data" << std::endl;
}


/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadBin(const std::string &file_path,
             const int num_dimensions,
             std::vector<std::vector<float>> &data) {
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  uint32_t N;  // num of points
  ifs.read((char *)&N, sizeof(uint32_t));
  data.resize(N);
  std::cout << "# of points: " << N << std::endl;
  std::vector<float> buff(num_dimensions);
  int counter = 0;
  while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float))) {
    std::vector<float> row(num_dimensions);
    for (int d = 0; d < num_dimensions; d++) {
      row[d] = static_cast<float>(buff[d]);
    }
    data[counter++] = std::move(row);
  }
  ifs.close();
  std::cout << "Finish Reading Data" << endl;
}


/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadKNN(const std::string &file_path,
             const int num_dimensions,
             std::vector<std::vector<unsigned>> &data) {
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  uint32_t N=10000;  // num of points
  // ifs.read((char *)&N, sizeof(uint32_t));
  data.resize(N);
  std::cout << "# of points: " << N << std::endl;
  std::vector<unsigned> buff(num_dimensions);
  std::cout << "num_dimensions = " << num_dimensions << "\n";
  int counter = 0;
  while (ifs.read((char *)buff.data(), num_dimensions * sizeof(unsigned))) {
    std::vector<unsigned> row(num_dimensions);
    for (int d = 0; d < num_dimensions; d++) {
      row[d] = static_cast<unsigned>(buff[d]);
    }
    data[counter++] = std::move(row);
  }
  ifs.close();
  std::cout << "Finish Reading Data" << endl;
}



char * LoadBin(const std::string &file_path,
             const int num_dimensions,
             size_t & row
            //  std::vector<std::vector<float>> &data
             ) {
  char * data{nullptr};
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  uint32_t N;  // num of points
  ifs.read((char *)&N, sizeof(uint32_t));
  // data.resize(N);
  std::cout << "# of points: " << N << std::endl;

  row = N;
  size_t col = num_dimensions;
  size_t stride = (sizeof(float) * col + KGRAPH_MATRIX_ALIGN - 1) / KGRAPH_MATRIX_ALIGN * KGRAPH_MATRIX_ALIGN;
  /*
  data.resize(row * stride);
  */
  if (data) free(data);
  data = (char *)memalign(KGRAPH_MATRIX_ALIGN, row * stride); // SSE instruction needs data to be aligned


  std::cout << "stride = " << stride << "\n";

  for (size_t i = 0; i < row; ++i) {
      ifs.read(&data[stride * i], sizeof(float) * col);
  }


  // std::vector<float> buff(num_dimensions);
  // int counter = 0;
  // while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float))) {
  //   std::vector<float> row(num_dimensions);
  //   for (int d = 0; d < num_dimensions; d++) {
  //     row[d] = static_cast<float>(buff[d]);
  //   }
  //   data[counter++] = std::move(row);
  // }
  ifs.close();
  std::cout << "Finish Reading Data" << endl;

  return data;
}


/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadKNN(const std::string &file_path,
             const int num_dimensions,
             std::vector<std::vector<unsigned>> &data, uint32_t N=10000) {
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  // uint32_t N=10000;  // num of points
  // ifs.read((char *)&N, sizeof(uint32_t));
  data.resize(N);
  std::cout << "# of points: " << N << std::endl;
  std::vector<unsigned> buff(num_dimensions);
  std::cout << "num_dimensions = " << num_dimensions << "\n";
  int counter = 0;
  while (ifs.read((char *)buff.data(), num_dimensions * sizeof(unsigned))) {
    std::vector<unsigned> row(num_dimensions);
    for (int d = 0; d < num_dimensions; d++) {
      row[d] = static_cast<unsigned>(buff[d]);
    }
    data[counter++] = std::move(row);
  }
  ifs.close();
  std::cout << "Finish Reading Data" << endl;
}

