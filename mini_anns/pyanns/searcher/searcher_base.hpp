#pragma once

#include <cstdint>

#include <queue>
#include <vector>

namespace pyanns {

using mini_pq = std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>>;

struct SearcherBase {
  virtual void SetData(const float *data, int32_t n, int32_t dim) = 0;
  virtual void Search(const float *q, int32_t k, int32_t *ids,
                      float *dis = nullptr) const = 0;
  virtual void Search2(const float *q, int32_t k, int32_t *ids,
                      // mini_pq & visit_pq, 
                      char * data, float l, float r, 
                      float *dis = nullptr) const = 0;
  virtual void Search3(const float *q, int32_t k, int32_t *ids,
                      // mini_pq & visit_pq, 
                      char * data, float l, float r, std::vector<uint32_t> & getid,
                      float *dis = nullptr) const = 0;
  virtual void SearchBatch(const float *q, int32_t nq, int32_t k, int32_t *ids,
                           float *dis = nullptr) const = 0;
  virtual ~SearcherBase() = default;
};

struct GraphSearcherBase : SearcherBase {
  virtual void SetEf(int32_t ef) = 0;
  virtual int32_t GetEf() const { return 0; }
  virtual void Optimize(int32_t num_threads = 0) = 0;
};

} // namespace pyanns
