#pragma once


#include <algorithm>
#include <mutex>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

#include <omp.h>

#include <mutex>

namespace rnndescent {


using LockGuard = std::lock_guard<std::mutex>;


inline void gen_random(std::mt19937& rng, int* addr, const int size, const int N) {
    for (int i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (int i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
            addr[i] = addr[i - 1] + 1;
        }
    }
    int off = rng() % N;
    for (int i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }
}

struct Neighbor {
    int id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(int id, float distance, bool f)
            : id(id), distance(distance), flag(f) {}

    inline bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

struct Nhood {
    std::mutex lock;
    std::vector<Neighbor> pool; // candidate pool (a max heap)
    int M;                      // number of new neighbors to be operated

    std::vector<int> nn_old;  // old neighbors
    std::vector<int> nn_new;  // new neighbors
    std::vector<int> rnn_old; // reverse old neighbors
    std::vector<int> rnn_new; // reverse new neighbors

    Nhood() = default;

    Nhood(int l, int s, std::mt19937& rng, int N) {
        M = s;
        nn_new.resize(s * 2);
        gen_random(rng, nn_new.data(), (int)nn_new.size(), N);
    }

    Nhood& operator=(const Nhood& other) {
        M = other.M;
        std::copy(
                other.nn_new.begin(),
                other.nn_new.end(),
                std::back_inserter(nn_new));
        nn_new.reserve(other.nn_new.capacity());
        pool.reserve(other.pool.capacity());
        return *this;
    }

    Nhood(const Nhood& other) {
        M = other.M;
        std::copy(
                other.nn_new.begin(),
                other.nn_new.end(),
                std::back_inserter(nn_new));
        nn_new.reserve(other.nn_new.capacity());
        pool.reserve(other.pool.capacity());
    }

    void insert(int id, float dist) {
        std::lock_guard<std::mutex> guard(lock);
        if (dist > pool.front().distance)
            return;
        for (int i = 0; i < pool.size(); i++) {
            if (id == pool[i].id)
                return;
        }
        if (pool.size() < pool.capacity()) {
            pool.push_back(Neighbor(id, dist, true));
            std::push_heap(pool.begin(), pool.end());
        } else {
            std::pop_heap(pool.begin(), pool.end());
            pool[pool.size() - 1] = Neighbor(id, dist, true);
            std::push_heap(pool.begin(), pool.end());
        }
    }

    template <typename C>
    void join(C callback) const {
        for (int const i : nn_new) {
            for (int const j : nn_new) {
                if (i < j) {
                    callback(i, j);
                }
            }
            for (int j : nn_old) {
                callback(i, j);
            }
        }
    }
};


// #pragma once 

// #include <algorithm>
// #include <mutex>
// #include <queue>
// #include <random>
// #include <unordered_set>
// #include <vector>

// #include <omp.h>

// #include <mutex>
// using LockGuard = std::lock_guard<std::mutex>;

// void gen_random(std::mt19937& rng, int* addr, const int size, const int N) {
//     for (int i = 0; i < size; ++i) {
//         addr[i] = rng() % (N - size);
//     }
//     std::sort(addr, addr + size);
//     for (int i = 1; i < size; ++i) {
//         if (addr[i] <= addr[i - 1]) {
//             addr[i] = addr[i - 1] + 1;
//         }
//     }
//     int off = rng() % N;
//     for (int i = 0; i < size; ++i) {
//         addr[i] = (addr[i] + off) % N;
//     }
// }



// struct Neighbor {
//     int id;
//     float distance;
//     bool flag;

//     Neighbor() = default;
//     Neighbor(int id, float distance, bool f)
//             : id(id), distance(distance), flag(f) {}

//     inline bool operator<(const Neighbor& other) const {
//         return distance < other.distance;
//     }
// };


// struct Nhood {
//     std::mutex lock;
//     std::vector<Neighbor> pool; // candidate pool (a max heap)
//     int M;                      // number of new neighbors to be operated

//     std::vector<int> nn_old;  // old neighbors
//     std::vector<int> nn_new;  // new neighbors
//     std::vector<int> rnn_old; // reverse old neighbors
//     std::vector<int> rnn_new; // reverse new neighbors

//     Nhood() = default;

//     Nhood(int l, int s, std::mt19937& rng, int N);

//     Nhood& operator=(const Nhood& other);

//     Nhood(const Nhood& other);

//     void insert(int id, float dist);

//     template <typename C>
//     void join(C callback) const;
// };


// enum MetricType {
//     METRIC_INNER_PRODUCT = 0, ///< maximum inner product search
//     METRIC_L2 = 1,            ///< squared L2 search
//     METRIC_L1,                ///< L1 (aka cityblock)
//     METRIC_Linf,              ///< infinity distance
//     METRIC_Lp,                ///< L_p distance, p is given by a faiss::Index
//                               /// metric_arg

//     /// some additional metrics defined in scipy.spatial.distance
//     METRIC_Canberra = 20,
//     METRIC_BrayCurtis,
//     METRIC_JensenShannon,
//     METRIC_Jaccard, ///< defined as: sum_i(min(a_i, b_i)) / sum_i(max(a_i, b_i))
//                     ///< where a_i, b_i > 0
// };


// Nhood::Nhood(int l, int s, std::mt19937& rng, int N) {
//     M = s;
//     nn_new.resize(s * 2);
//     gen_random(rng, nn_new.data(), (int)nn_new.size(), N);
// }

// /// Copy operator
// Nhood& Nhood::operator=(const Nhood& other) {
//     M = other.M;
//     std::copy(
//             other.nn_new.begin(),
//             other.nn_new.end(),
//             std::back_inserter(nn_new));
//     nn_new.reserve(other.nn_new.capacity());
//     pool.reserve(other.pool.capacity());
//     return *this;
// }

// /// Copy constructor
// Nhood::Nhood(const Nhood& other) {
//     M = other.M;
//     std::copy(
//             other.nn_new.begin(),
//             other.nn_new.end(),
//             std::back_inserter(nn_new));
//     nn_new.reserve(other.nn_new.capacity());
//     pool.reserve(other.pool.capacity());
// }

// /// Insert a point into the candidate pool
// void Nhood::insert(int id, float dist) {
//     LockGuard guard(lock);
//     if (dist > pool.front().distance)
//         return;
//     for (int i = 0; i < pool.size(); i++) {
//         if (id == pool[i].id)
//             return;
//     }
//     if (pool.size() < pool.capacity()) {
//         pool.push_back(Neighbor(id, dist, true));
//         std::push_heap(pool.begin(), pool.end());
//     } else {
//         std::pop_heap(pool.begin(), pool.end());
//         pool[pool.size() - 1] = Neighbor(id, dist, true);
//         std::push_heap(pool.begin(), pool.end());
//     }
// }

// /// In local join, two objects are compared only if at least
// /// one of them is new.
// template <typename C>
// void Nhood::join(C callback) const {
//     for (int const i : nn_new) {
//         for (int const j : nn_new) {
//             if (i < j) {
//                 callback(i, j);
//             }
//         }
//         for (int j : nn_old) {
//             callback(i, j);
//         }
//     }
// }
}