#pragma once

#include "pyanns/common.hpp"
#include "pyanns/graph.hpp"
#include "pyanns/neighbor.hpp"
#include "pyanns/quant/product_quant.hpp"
#include "pyanns/quant/quant.hpp"
#include "pyanns/quant/quant_base.hpp"
#include "pyanns/searcher/refiner.hpp"
#include "pyanns/searcher/searcher_base.hpp"
#include "pyanns/utils.hpp"

#include "pyanns/searcher/visited_list_pool.h"
// #include "pyanns/searcher/metrics.hpp"

#include <algorithm>
#include <omp.h>
#include <random>

// #include "/home/jfwang/proj/SSS/helper/learn_pyanns/mini_anns/utils/IO.hpp" ///FIXME: debug use

namespace pyanns
{

    namespace params
    {

        constexpr inline bool SQ8_REFINE = true;
        constexpr inline bool SQ8U_REFINE = true;
        constexpr inline bool SQ8P_REFINE = true;
        constexpr inline bool SQ4U_REFINE = true;
        constexpr inline bool SQ4UA_REFINE = true;
        constexpr inline bool PQ8_REFINE = true;

        constexpr inline int32_t SQ8_REFINE_FACTOR = 10;
        constexpr inline int32_t SQ8U_REFINE_FACTOR = 2;
        constexpr inline int32_t SQ8P_REFINE_FACTOR = 10;
        constexpr inline int32_t SQ4U_REFINE_FACTOR = 10;
        constexpr inline int32_t SQ4UA_REFINE_FACTOR = 10;
        constexpr inline int32_t PQ8_REFINE_FACTOR = 10;

        template <Metric metric>
        // using RefineQuantizer = FP16Quantizer<metric>;
        using RefineQuantizer = FP32Quantizer<metric>; /// jfwang-mod
        

    } // namespace params

    inline uint64_t xor_genrand(uint64_t x)
    {
        x ^= x >> 12; // a
        x ^= x << 25; // b
        x ^= x >> 27; // c
        return x * 0x2545F4914F6CDD1D;
    }

    struct Comparerrr {
        constexpr bool operator()(std::pair<float, int> const& a,
            std::pair<float, int> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    template <QuantConcept Quant>
    struct GraphSearcher : public GraphSearcherBase
    {

        int32_t d;
        int32_t nb;
        Graph<int32_t> graph;
        Quant quant;

        // Search parameters
        int32_t ef = 32;

        // Memory prefetch parameters
        int32_t po = 1;
        int32_t pl = 1;
        int32_t graph_po = 1;
        VisitedListPool *visited_list_pool_{nullptr};

        // Optimization parameters
        constexpr static int32_t kOptimizePoints = 1000;
        constexpr static int32_t kTryPos = 10;
        constexpr static int32_t kTryPls = 10;
        constexpr static int32_t kTryK = 10;
        int32_t sample_points_num;
        std::vector<float> optimize_queries;

        GraphSearcher(Graph<int32_t> g)
            : graph(std::move(g)), graph_po(graph.K / 16)
        {
            visited_list_pool_ = new VisitedListPool(32, graph.N);
        }

        ~GraphSearcher()
        {
            delete visited_list_pool_;
        }

        void SetData(const float *data, int32_t n, int32_t dim) override
        {
            this->nb = n;
            this->d = dim;
            quant = Quant(d);
            // printf("Starting quantizer training\n");
            // auto t1 = std::chrono::high_resolution_clock::now();
            quant.train(data, n);
            quant.add(data, n);
            // auto t2 = std::chrono::high_resolution_clock::now();
            // printf("Done quantizer training, cost %.2lfs\n",
            //        std::chrono::duration<double>(t2 - t1).count());

            sample_points_num = std::min(kOptimizePoints, nb - 1);
            std::vector<int32_t> sample_points(sample_points_num);
            std::mt19937 rng;
            GenRandom(rng, sample_points.data(), sample_points_num, nb);
            optimize_queries.resize((int64_t)sample_points_num * d);
            for (int32_t i = 0; i < sample_points_num; ++i)
            {
                memcpy(optimize_queries.data() + (int64_t)i * d,
                       data + (int64_t)sample_points[i] * d, d * sizeof(float));
            }
        }

        void SetEf(int32_t ef) override { this->ef = ef; }

        int32_t GetEf() const override { return ef; }

        void Optimize(int32_t = 0) override
        {
            std::vector<int32_t> try_pos(std::min(kTryPos, graph.K));
            std::vector<int32_t> try_pls(
                std::min(kTryPls, (int32_t)upper_div(quant.code_size(), 64)));
            std::iota(try_pos.begin(), try_pos.end(), 1);
            std::iota(try_pls.begin(), try_pls.end(), 1);
            std::vector<int32_t> dummy_dst(kTryK);

            auto f = [&]
            {
#pragma omp parallel for schedule(dynamic)
                for (int32_t i = 0; i < sample_points_num; ++i)
                {
                    Search(optimize_queries.data() + (int64_t)i * d, kTryK,
                           dummy_dst.data());
                }
            };
            // printf("=============Start optimization=============\n");
            // warmup
            f();
            float min_ela = std::numeric_limits<float>::max();
            int32_t best_po = 0, best_pl = 0;
            for (auto try_po : try_pos)
            {
                for (auto try_pl : try_pls)
                {
                    this->po = try_po;
                    this->pl = try_pl;
                    auto st = std::chrono::high_resolution_clock::now();
                    f();
                    auto ed = std::chrono::high_resolution_clock::now();
                    auto ela = std::chrono::duration<double>(ed - st).count();
                    if (ela < min_ela)
                    {
                        min_ela = ela;
                        best_po = try_po;
                        best_pl = try_pl;
                    }
                }
            }
            float baseline_ela;
            {
                this->po = 1;
                this->pl = 1;
                auto st = std::chrono::high_resolution_clock::now();
                f();
                auto ed = std::chrono::high_resolution_clock::now();
                baseline_ela = std::chrono::duration<double>(ed - st).count();
            }
            float slow_ela;
            {
                this->po = 0;
                this->pl = 0;
                auto st = std::chrono::high_resolution_clock::now();
                f();
                auto ed = std::chrono::high_resolution_clock::now();
                slow_ela = std::chrono::duration<double>(ed - st).count();
            }

            // printf("settint best po = %d, best pl = %d\n"
            //        "gaining %6.2f%% performance improvement wrt baseline\ngaining "
            //        "%6.2f%% performance improvement wrt slow\n============="
            //        "Done optimization=============\n",
            //        best_po, best_pl, 100.0 * (baseline_ela / min_ela - 1),
            //        100.0 * (slow_ela / min_ela - 1));
            this->po = best_po;
            this->pl = best_pl;
            std::vector<float>().swap(optimize_queries);
        }

        void SearchImpl(inference::NeighborPoolConcept auto &pool,
                        ComputerConcept auto &computer) const
        {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

            alignas(64) int32_t edge_buf[graph.K];
            while (pool.has_next())
            {
                auto u = pool.pop();
                graph.prefetch(u, graph_po);
                int32_t edge_size = 0;
                for (int32_t i = 0; i < graph.K; ++i)
                {
                    int32_t v = graph.at(u, i);
                    if (v == -1)
                    {
                        break;
                    }
                    // if (pool.is_visited(v))
                    // {
                    //     continue;
                    // }
                    // pool.set_visited(v);
                    if (visited_array[v] == visited_array_tag)
                    {
                        continue;
                    }
                    visited_array[v] = visited_array_tag;

                    edge_buf[edge_size++] = v;
                }
                for (int i = 0; i < std::min(po, edge_size); ++i)
                {
                    computer.prefetch(edge_buf[i], pl);
                }
                for (int i = 0; i < edge_size; ++i)
                {
                    if (i + po < edge_size)
                    {
                        computer.prefetch(edge_buf[i + po], pl);
                    }
                    auto v = edge_buf[i];
                    auto cur_dist = computer(v);
                    pool.insert(v, cur_dist);
                }
            }
        visited_list_pool_->releaseVisitedList(vl);
        }
        void Search(const float *q, int32_t k, int32_t *ids,
                    float *dis = nullptr) const override
        {
            auto computer = quant.get_computer(q);
            inference::LinearPool<typename Quant::ComputerType::dist_type> pool(
                nb, std::max(k, ef), k);
            graph.initialize_search(pool, computer);
            SearchImpl(pool, computer);
            for (int32_t i = 0; i < k; ++i)
            {
                ids[i] = pool.id(i);
                if (dis != nullptr)
                {
                    dis[i] = pool.dist(i);
                }
            }
        }

        void SearchImpl2(inference::NeighborPoolConcept auto &pool,
                         ComputerConcept auto &computer,
                        //  mini_pq &visit_pq,
                         char *data, float l, float r,
                         inference::NeighborPoolConcept auto &visit_pool
                         ) const
        {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

            alignas(64) int32_t edge_buf[graph.K];
            while (pool.has_next())
            {
                auto u = pool.pop();
                graph.prefetch(u, graph_po);
                int32_t edge_size = 0;
                for (int32_t i = 0; i < graph.K; ++i)
                {
                    int32_t v = graph.at(u, i);
                    if (v == -1)
                    {
                        break;
                    }
                    // if (pool.is_visited(v))
                    // {
                    //     continue;
                    // }
                    // pool.set_visited(v);
                    if (visited_array[v] == visited_array_tag)
                    {
                        continue;
                    }
                    visited_array[v] = visited_array_tag;
                    edge_buf[edge_size++] = v;
                }
                for (int i = 0; i < std::min(po, edge_size); ++i)
                {
                    computer.prefetch(edge_buf[i], pl);
                }
                for (int i = 0; i < edge_size; ++i)
                {
                    if (i + po < edge_size)
                    {
                        computer.prefetch(edge_buf[i + po], pl);
                    }
                    auto v = edge_buf[i];
                    auto cur_dist = computer(v);
                    pool.insert(v, cur_dist);

                    {
                        auto data_id = (float *)(data + size_t(v) * 416);
                        auto ts_id = (data_id[1]);
                        if (ts_id >= l && ts_id <= r){
                            visit_pool.insert(v, cur_dist);
                        }
                    }
                }
            }
        visited_list_pool_->releaseVisitedList(vl);
        }

        void SearchImpl2HNSW(inference::NeighborPoolConcept auto &pool,
                             ComputerConcept auto &computer,
                             float *data_point,
                             int efsrch,
                             mini_pq &visit_pq,
                             char *data, float l, float r) const
        {
//             auto knn_graph = IO::LoadBinVec<unsigned>("/home/jfwang/proj/SSS/running_graph/running_test_rnn_10m_index_d100.ivecs");

//             VisitedList *vl = visited_list_pool_->getFreeVisitedList();
//             vl_type *visited_array = vl->mass;
//             vl_type visited_array_tag = vl->curV;

//             // std::cout << "nb = " << nb << "\n";
//             // std::cout << "efsrch = " << efsrch << "\n";
//             std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, Comparerrr> top_candidates;
//             std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, Comparerrr> candidate_set;

//             auto lowerBound = std::numeric_limits<float>::max();
// // std::cout << "graph.size() = " << knn_graph.size() << "\n";
// // std::cout << "l = " << l << ", r = " << r << "\n";
// // std::cout << "show random seeds:\n";
//             for (int i = 0; i < 32; ++i)
//             {
//                 int idx = xor_genrand(i + 1) % (nb);
//                 if (visited_array[idx] == visited_array_tag)
//                 {
//                     continue;
//                 }
//                 visited_array[idx] = visited_array_tag;
//                 float dist = basicL2((char *)data_point, data + size_t(idx) * 416 + 8, 100);
// // std::cout << idx << ":" << dist << ", ";

//                 candidate_set.emplace(-dist, idx);
//                 top_candidates.emplace(dist, idx);

//                 auto data_id = (float *)(data + size_t(idx) * 416);
//                 auto ts_id = (data_id[1]);

//                 if (ts_id >= l && ts_id <= r)
//                     visit_pq.emplace(dist, idx);

//                 while (top_candidates.size() > efsrch)
//                 {
//                     top_candidates.pop();
//                 }
//             }
// // std::cout <<  "\n";

//             // std::cout <<  " with SearchImpl2HNSW\n";
//             // std::cout << "at present, visit_pq.size = " << visit_pq.size() << "\n";
//             // {
//             //     // std::cout << "show visit_pq:\n";
//             //     auto copy = visit_pq;
//             //     while (copy.size()) {
//             //         std::cout << copy.top().first << " - " << copy.top().second << "\n";
//             //         copy.pop();
//             //     }
//             // }

//             unsigned iter = 0;
//             while (!candidate_set.empty())
//             {
//                 std::pair<float, int> current_node_pair = candidate_set.top();
//                 if ((-current_node_pair.first) > lowerBound &&
//                     (top_candidates.size() == efsrch))
//                 { 
//                     break;
//                 }
//                 candidate_set.pop();
//                 int current_node_id = current_node_pair.second;

//                 auto &nbhood = knn_graph[current_node_id];
//                 size_t size = nbhood.size();

//                 for (unsigned j = 0; j < size; ++j)
//                 {

//                     auto candidate_id = nbhood[j];

//                     if (!(visited_array[candidate_id] == visited_array_tag))
//                     {
//                         visited_array[candidate_id] = visited_array_tag;
//                         float dist = basicL2((char *)data_point, data + size_t(candidate_id) * 416 + 8, 100);

//                         auto data_id = (float *)(data + size_t(candidate_id) * 416);
//                         auto ts_id = (data_id[1]);

//                         if (ts_id >= l && ts_id <= r)
//                             visit_pq.emplace(dist, candidate_id);

//                         if (top_candidates.size() < ef || lowerBound > dist)
//                         {
//                             candidate_set.emplace(-dist, candidate_id);
//                             // _mm_prefetch(data + (candidate_set.top().second) * 416 + DATA_OFFSET, _MM_HINT_T0);
//                             top_candidates.emplace(dist, candidate_id);

//                             if (top_candidates.size() > ef)
//                                 top_candidates.pop();

//                             if (!top_candidates.empty())
//                                 lowerBound = top_candidates.top().first;
//                         }
//                     }
//                 }
//                 while (visit_pq.size() > 100)
//                 {
//                     visit_pq.pop();
//                 }
//             }

//             // std::cout << "at present@@, visit_pq.size = " << visit_pq.size() << "\n";
//             // {
//             //     std::cout << "show visit_pq:\n";
//             //     auto copy = visit_pq;
//             //     while (copy.size() > 100)
//             //     {
//             //         copy.pop();
//             //     }
//             //     while (copy.size())
//             //     {
//             //         std::cout << copy.top().first << " - " << copy.top().second << "\n";
//             //         copy.pop();
//             //     }
//             // }

//             visited_list_pool_->releaseVisitedList(vl);
        }

            // SearchImpl2HNSW(pool, computer, const_cast<float *>(q), std::max(k, ef), visit_pq, data, l, r);
        void Search2(const float *q, int32_t k, int32_t *ids,
                    //  mini_pq &visit_pq,
                     char *data, float l, float r,
                     float *dis = nullptr) const
        {
            // auto computer = quant.get_computer(q);
            // inference::LinearPool<typename Quant::ComputerType::dist_type> pool(
            //     nb, std::max(k, ef), k);
            // graph.initialize_search(pool, computer);
            // SearchImpl(pool, computer);
            // for (int32_t i = 0; i < k; ++i)
            // {
            //     ids[i] = pool.id(i);
            //     if (dis != nullptr)
            //     {
            //         dis[i] = pool.dist(i);
            //     }
            // }
            /////old
            auto computer = quant.get_computer(q);
            inference::LinearPool<typename Quant::ComputerType::dist_type> pool(
                nb, std::max(k, ef), k);
            // inference::LinearPool<typename Quant::ComputerType::dist_type> visit_pool(
            //     nb, std::max(k, ef), k);
            inference::LinearPool<typename Quant::ComputerType::dist_type> visit_pool(
                nb, k, k);
            graph.initialize_search(pool, computer);
            // SearchImpl2(pool, computer, visit_pq, data, l, r);
            SearchImpl2(pool, computer, data, l, r, visit_pool);

            // while (visit_pq.size() > k)
            // { //// FP32 directly
            //     visit_pq.pop();
            // }
            // for (int32_t i = 0; i < k; ++i)
            // {
            //     ids[i] = visit_pq.top().second;
            //     visit_pq.pop();
            //     // ids[i] = pool.id(i);
            //     if (dis != nullptr)
            //     {
            //         dis[i] = pool.dist(i);
            //     }
            // }

            for (int32_t i = 0; i < k; ++i)//// SQ8
            {
                // ids[i] = pool.id(i);
                // if (dis != nullptr)
                // {
                //     dis[i] = pool.dist(i);
                // }
                ids[i] = visit_pool.id(i);
                if (dis != nullptr)
                {
                    dis[i] = visit_pool.dist(i);
                }
            }
        }



        void SearchImpl3(inference::NeighborPoolConcept auto &pool,
                         ComputerConcept auto &computer,
                        inference::NeighborPoolConcept auto &visit_pool,
                         char *data, float l, float r, std::vector<uint32_t> & getid) const
        {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

            alignas(64) int32_t edge_buf[graph.K];
            while (pool.has_next())
            {
                auto u = pool.pop();
                graph.prefetch(u, graph_po);
                int32_t edge_size = 0;
                for (int32_t i = 0; i < graph.K; ++i)
                {
                    int32_t v = graph.at(u, i);
                    if (v == -1)
                    {
                        break;
                    }
                    // if (pool.is_visited(v))
                    // {
                    //     continue;
                    // }
                    // pool.set_visited(v);
                    if (visited_array[v] == visited_array_tag)
                    {
                        continue;
                    }
                    visited_array[v] = visited_array_tag;

                    edge_buf[edge_size++] = v;
                }
                for (int i = 0; i < std::min(po, edge_size); ++i)
                {
                    computer.prefetch(edge_buf[i], pl);
                }
                for (int i = 0; i < edge_size; ++i)
                {
                    if (i + po < edge_size)
                    {
                        computer.prefetch(edge_buf[i + po], pl);
                    }
                    auto v = edge_buf[i];
                    auto cur_dist = computer(v);
                    pool.insert(v, cur_dist);

                    {
                        auto data_id = (float *)(data + getid[size_t(v)] * 416); ////FIXME: 
                        auto ts_id = (data_id[1]);
                        if (ts_id >= l && ts_id <= r)
                            visit_pool.insert(v, cur_dist);
                            // visit_pq.emplace(cur_dist, v); /// FIXME: still v for refine distance calculation
                    }
                }
            }
        visited_list_pool_->releaseVisitedList(vl);
        }


        void Search3(const float *q, int32_t k, int32_t *ids,
                    //  mini_pq &visit_pq,
                     char *data, float l, float r, std::vector<uint32_t> & getid,
                     float *dis = nullptr) const
        {
            auto computer = quant.get_computer(q);
            inference::LinearPool<typename Quant::ComputerType::dist_type> pool(
                nb, std::max(k, ef), k);
            // inference::LinearPool<typename Quant::ComputerType::dist_type> visit_pool(
            //     nb, std::max(k, ef), k);
            inference::LinearPool<typename Quant::ComputerType::dist_type> visit_pool(
                nb, k, k);
            graph.initialize_search(pool, computer);

            SearchImpl3(pool, computer, visit_pool, data, l, r, getid);

            // while (visit_pq.size() > k)
            // { //// FP32 directly
            //     visit_pq.pop();
            // }
            // for (int32_t i = 0; i < k; ++i)
            // {
            //     ids[i] = visit_pq.top().second;
            //     visit_pq.pop();
            //     // ids[i] = pool.id(i);
            //     if (dis != nullptr)
            //     {
            //         dis[i] = pool.dist(i);
            //     }
            // }

            for (int32_t i = 0; i < k; ++i)//// SQ8
            {
                // ids[i] = pool.id(i);
                // if (dis != nullptr)
                // {
                //     dis[i] = pool.dist(i);
                // }
                ids[i] = visit_pool.id(i);
                if (dis != nullptr)
                {
                    dis[i] = visit_pool.dist(i);
                }
            }
        }


        void SearchBatch(const float *q, int32_t nq, int32_t k, int32_t *ids,
                         float *dis = nullptr) const override
        {
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < nq; ++i)
            {
                Search(q + i * d, k, ids + i * k, dis ? dis + i * k : nullptr);
            }
        }
    };

    inline std::unique_ptr<GraphSearcherBase>
    create_searcher(Graph<int32_t> graph, const std::string &metric,
                    const std::string &quantizer = "FP16")
    {
        using RType = std::unique_ptr<GraphSearcherBase>;
        auto m = metric_map[metric];
        auto qua = quantizer_map[quantizer];

        if (qua == QuantizerType::SQ8U)
        {
            if (m == Metric::IP)
            {
                RType ret =
                    std::make_unique<GraphSearcher<SQ8QuantizerUniform<Metric::IP>>>(
                        std::move(graph));
                if (params::SQ8U_REFINE)
                {
                    ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::IP>>>(
                        std::move(ret), params::SQ8U_REFINE_FACTOR);
                }
                return ret;
            }
            else if (m == Metric::L2)
            {
                RType ret =
                    std::make_unique<GraphSearcher<SQ8QuantizerUniform<Metric::L2>>>(
                        std::move(graph));
                if (params::SQ8U_REFINE)
                {
                    ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::L2>>>(
                        std::move(ret), params::SQ8U_REFINE_FACTOR);
                }
                return ret;
            }
            else
            {
                printf("Metric not suppported\n");
                return nullptr;
            }
        }
        else if (qua == QuantizerType::FP32)
        {
            if (m == Metric::IP)
            {
                RType ret = std::make_unique<GraphSearcher<FP32Quantizer<Metric::IP>>>(
                    std::move(graph));
                return ret;
            }
            else
            {
                RType ret = std::make_unique<GraphSearcher<FP32Quantizer<Metric::L2>>>(
                    std::move(graph));
                return ret;
            }
        }
        else if (qua == QuantizerType::FP16)
        {
            if (m == Metric::IP)
            {
                RType ret = std::make_unique<GraphSearcher<FP16Quantizer<Metric::IP>>>(
                    std::move(graph));
                return ret;
            }
            else
            {
                RType ret = std::make_unique<GraphSearcher<FP16Quantizer<Metric::L2>>>(
                    std::move(graph));
                return ret;
                // printf("Metric not suppported\n");
                // return nullptr;
            }
        }
        else if (qua == QuantizerType::SQ4U)
        {
            if (m == Metric::IP)
            {
                RType ret =
                    std::make_unique<GraphSearcher<SQ4QuantizerUniform<Metric::IP>>>(
                        std::move(graph));
                if (params::SQ4U_REFINE)
                {
                    ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::IP>>>(
                        std::move(ret), params::SQ4U_REFINE_FACTOR);
                }
                return ret;
            }
            else if (m == Metric::L2)
            {
                RType ret =
                    std::make_unique<GraphSearcher<SQ4QuantizerUniform<Metric::L2>>>(
                        std::move(graph));
                if (params::SQ4U_REFINE)
                {
                    ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::L2>>>(
                        std::move(ret), params::SQ4U_REFINE_FACTOR);
                }
                return ret;
            }
            else
            {
                printf("Metric not suppported\n");
                return nullptr;
            }
        }
        else
        {
            printf("Quantizer type not supported\n");
            return nullptr;
        }
    }

} // namespace pyanns
