/**
 *  Example code using sampling to find KNN.
 *
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <chrono>
#include "io.h"
#include "utils/metrics.hpp"
#include "utils/nnvertex.hpp"
#include "utils/tobedelio.hpp"
#include <cmath>

#include <omp.h>
#include <unordered_map>


#include "mini_rnn/RNNDescent.h"
#include "mini_anns/pyanns/searcher/graph_searcher.hpp"


#include <memory>



using std::cout;
using std::endl;
using std::string;
using std::vector;

using namespace std;

#define DATA_OFFSET 8
#define QUERY_OFFSET 16
#define BUILD_GRAPH_THRES 5000

#define RATIO 5
#define NGPS 10

#define RANGE 0

#define CHUNK 100

using graph_t = std::vector<std::vector<unsigned>>;
using mapList = std::unordered_map<unsigned, std::vector<unsigned>>;
using mapGraph = std::unordered_map<unsigned, graph_t>;


float getRecallMini(std::vector<std::vector<unsigned>> &anng, std::vector<std::vector<unsigned>> &knng, std::vector<unsigned> & list, size_t checkK)
{
    size_t hit = 0;
    size_t checkN = std::min(anng.size(), knng.size());

    // for (size_t i = 0; i < checkN; ++i)
    for (unsigned i : list)
    {
        auto &ann = anng[i];
        auto &knn = knng[i];
        for (size_t j = 0; j < checkK; ++j)
        {
            auto idx = ann[j];
            for (size_t l = 0; l < checkK; ++l)
            {
                auto nb = knn[l];
                if (idx == nb)
                {
                    ++hit;
                    break;
                }
            }
        }
    }
    return 1.0 * hit / (checkK * list.size());
}


    inline uint64_t GenerateRandomNumber(uint64_t x) {
        x ^= x >> 12;  // a
        x ^= x << 25;  // b
        x ^= x >> 27;  // c
        return x * 0x2545F4914F6CDD1D;
    }



void stat(const vector<vector<unsigned>> &graph)
{
    size_t max_edge = 0;
    size_t min_edge = graph.size();
    size_t avg_edge = 0;
    for (auto &nbhood : graph)
    {
        auto size = nbhood.size();
        max_edge = std::max(max_edge, size);
        min_edge = std::min(min_edge, size);
        avg_edge += size;
    }
    std::cout << "max_edge = " << max_edge << "\nmin_edge = " << min_edge << "\navg_edge = " << (1.0 * avg_edge / graph.size()) << "\n";
}


    using dist_t = float; 
    using tableint = unsigned int; 
    // using namespace hnswlib::hnswl;

    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };








std::pair<unsigned, unsigned>
findRangePositions(std::vector<std::pair<float, unsigned>>& sortedPairs, float left, float right) {
    // Find the position of the first element greater than or equal to left
    auto leftPos = std::lower_bound(sortedPairs.begin(), sortedPairs.end(), left,
        [](const std::pair<float, unsigned>& pair, float value) {
            return pair.first < value;
        });

    // Find the position of the first element greater than right
    auto rightPos = std::upper_bound(sortedPairs.begin(), sortedPairs.end(), right,
        [](float value, const std::pair<float, unsigned>& pair) {
            return value < pair.first;
        });

    // Calculate the index of leftPos and rightPos
    unsigned leftIndex = std::distance(sortedPairs.begin(), leftPos);
    unsigned rightIndex = std::distance(sortedPairs.begin(), rightPos);

    return {leftIndex, rightIndex};
}


unsigned int getRangeIndex(float range) {
    // Ensure range is within [0, 1]
    range = std::max(0.0f, std::min(1.0f, range));

    // Calculate the index based on the range
    unsigned int index = static_cast<unsigned int>(range * 10);

    return index;
}


inline 
vector<unsigned> 
BFSearch(std::vector<std::pair<float, unsigned>>& sortedPairs, unsigned lidx, unsigned ridx, 
char *data, char *query
) {
        vector<NNItem> nnlist;
        vector<unsigned> nn_idx(100);
        for (unsigned i = lidx; i <= ridx; ++i)
        {
            auto j = sortedPairs[i].second;
            auto dst = basicL2((float *)(data + j * 416 + DATA_OFFSET), (float*)(query + QUERY_OFFSET), 100);
            // auto dst = fstdistfunc_((float *)(data + j * 416 + DATA_OFFSET), (float*)(query + QUERY_OFFSET), dist_func_param_);
            nnlist.emplace_back(j, dst);
        }
        std::sort(nnlist.begin(), nnlist.end());
        for (size_t iter = 0; iter < 100 && iter < nnlist.size(); ++iter)
        {
            nn_idx[iter] = (nnlist[iter].idx);
        }
        return nn_idx;
}


inline 
vector<unsigned> 
BFSearchXX(std::vector<std::pair<float, unsigned>>& sortedPairs, unsigned lidx, unsigned ridx, 
char *data, char *query, unsigned catgory
) {

        // auto qq = (float *)(query);
        // float v = qq[1];
        // float l = qq[2];
        // float r = qq[3];

        // float qt = qq[0];

        vector<NNItem> nnlist;
        vector<unsigned> nn_idx(100);

        // unsigned pre_size = (ridx - lidx + 1);
        // unsigned hit{0};

        for (unsigned i = lidx; i <= ridx; ++i)
        {
            auto j = sortedPairs[i].second;
            auto this_data = (float *) (data + j * 416);
            auto c = unsigned(this_data[0]); 
            auto val = this_data[1];

            // if (val >= l && val <= r)
            //     ++hit;

            if (c != catgory) continue;

            auto dst = basicL2((float *)(data + j * 416 + DATA_OFFSET), (float*)(query + QUERY_OFFSET), 100);
            // auto dst = fstdistfunc_((float *)(data + j * 416 + DATA_OFFSET), (float*)(query + QUERY_OFFSET), dist_func_param_);

            nnlist.emplace_back(j, dst);
        }
        // printf("psize = %d - actu size = %d\n", pre_size, hit);
        std::sort(nnlist.begin(), nnlist.end());
        for (size_t iter = 0; iter < 100 && iter < nnlist.size(); ++iter)
        {
            nn_idx[iter] = (nnlist[iter].idx);
        }
        return nn_idx;
}


vector<unsigned> extreme_fast_get_nbhoodLarge(std::unordered_map<unsigned, unsigned> &id2size,
mapList &id2list, mapGraph & id2graph, graph_t & knn_graph,
    char *data, size_t data_size, size_t data_dim, char *query,
    mapList & dis2list,  mapGraph & dis2graph,
    vector<float> & ratio_list,
    vector<float> & ratio_list_lvl2,
    std::vector<std::pair<float, unsigned>> & data_pair,
    std::unique_ptr<pyanns::GraphSearcherBase> & nns_searcher,
    std::unordered_map<unsigned, std::unique_ptr<pyanns::GraphSearcherBase>> & id2srcher,
     const unsigned KK = 100)
{
    vector<unsigned> nn_idx;

    float this_ratio = 3;
    float qt2_ef = 1000;

    bool bbf{false};

    auto qq = (float *)(query);
    float v = qq[1];
    float l = qq[2];
    float r = qq[3];

    float qt = qq[0];

    // if (qt > 1) { ////NICE
    //     auto ridx = getRangeIndex(r - l);
    //     this_ratio = ratio_list[ridx];
    //     if (ridx == 0 && qt == 2) { // range 0 - 0.1 && just for qt2
    //         auto ridx = getRangeIndex(10 * (r - l));
    //         this_ratio = ratio_list_lvl2[ridx];
    //         if (ridx == 0) { // range 0 - 0.01 
    //             bbf = true;
    //         }
    //     }
    // }

    if (qt > 1) {
        auto ridx = getRangeIndex(r - l);
        this_ratio = ratio_list[ridx];
        if (ridx == 0) { // range 0 - 0.1 && just for qt2
            if (qt == 2){
                auto ridx = getRangeIndex(10 * (r - l));
                this_ratio = ratio_list_lvl2[ridx];
                // if (ridx == 0) { // range 0 - 0.01 
                if (ridx < 2) { // range 0 - 0.01 
                    bbf = true;
                }
            }else{//qt=3
                auto ridx = getRangeIndex(10 * (r - l));
                // this_ratio = 7;
                this_ratio = ratio_list_lvl2[ridx];
                if (ridx < 2) { // range 0 - 0.01 
                    bbf = true; //// top3 ie < 3 all can bbf 
                }
            }
        }
    }



    if (qt == 0)
    {
        {
            nns_searcher->SetEf(2300);////good
            // nns_searcher->SetEf(2600);////good2
            // nns_searcher->SetEf(2500);////

            // nns_searcher->SetEf(3500);
            // nns_searcher->SetEf(3100);
            // nns_searcher->SetEf(2800);
            // nns_searcher->SetEf(3200);
            auto srched_res = new int[KK];
            nns_searcher->Search((float*)(query + QUERY_OFFSET), KK, srched_res);

            for (unsigned i = 0; i < KK; ++i) {
                nn_idx.emplace_back((unsigned)(srched_res[i]));
            }

            delete [] srched_res;

            // auto results = searchSingleBaseLayerX(0, query + QUERY_OFFSET, 100 * RATIO * 1.5, visited_list_pool_, start_points_, initial_seed_size, data, knn_graph);
            // while (results.size() > KK){results.pop();}
            // while (results.size()) {
            //     nn_idx.emplace_back(results.top().second);
            //     results.pop();
            // }
            // reverse(nn_idx.begin(), nn_idx.end());

        }
        return nn_idx;
    }
    else if (qt == 1)
    {
        {


            // vector<unsigned> nn_idx;

            auto catgory = (unsigned)(v);
            auto its_size = id2size[catgory];
            // cout << "its_size = " << its_size << "\n";
            // exit(0);
            if (its_size >= BUILD_GRAPH_THRES) {
                // auto & minigraph = id2graph[catgory];

                // auto results = searchSingleBaseLayerM(0, query + QUERY_OFFSET, 100 * RATIO * 1.7, visited_list_pool_, data, minigraph, id2list[catgory]);

                // while (results.size() > KK){results.pop();}
                // while (results.size()) {
                //     nn_idx.emplace_back(id2list[catgory][results.top().second]);
                //     results.pop();
                // }
                // reverse(nn_idx.begin(), nn_idx.end());

                auto & mini_searcher = id2srcher[catgory];
                mini_searcher->SetEf(2000);
                // mini_searcher->SetEf((int)(2600 * std::pow(1.0 * its_size / knn_graph.size(), 0.2)));
                auto srched_res = new int[KK];
                mini_searcher->Search((float*)(query + QUERY_OFFSET), KK, srched_res);

                for (unsigned i = 0; i < KK; ++i) {
                    nn_idx.emplace_back(id2list[catgory][(unsigned)(srched_res[i])]);
                }

                delete [] srched_res;

            } else {
                auto & this_list = id2list[catgory];
                vector<NNItem> nnlist;

                for (size_t j = 0; j < its_size; ++j)
                {
                    // auto node = (float *)(data + j * 416);
                    // // std::cout << "C:" << node[0] << " T:" << node[1] << "\n";
                    // if (node[0] != v)
                    //     continue;
                    auto n_id = this_list[j]; 

                    auto dst = basicL2(data + n_id * 416 + DATA_OFFSET, query + QUERY_OFFSET, data_dim);
                    // auto dst = fstdistfunc_(data + n_id * 416 + DATA_OFFSET, query + QUERY_OFFSET, dist_func_param_);

                    nnlist.emplace_back(n_id, dst);
                }
                std::sort(nnlist.begin(), nnlist.end());
                for (size_t iter = 0; iter < KK && iter < nnlist.size(); ++iter)
                {
                    nn_idx.emplace_back(nnlist[iter].idx);
                }
                if (nn_idx.size() < KK) {
                    // cout << "compos nbhoods:\n";
                    unsigned id{0};
                    while (nn_idx.size() < KK) {
                        nn_idx.emplace_back(id++);
                    }
                }
            }


        }
        // bbf = true;

        return nn_idx;
    }
    else if (qt == 2)
    {

        if (bbf) { /// true
            auto [lp, rp] = findRangePositions(data_pair, l, r) ;
            return BFSearch(data_pair, lp, rp, data, query);
        }
        else
        {
           {///// new
                // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> visit_pq;
                // searchSingleBaseLayerXXXX(0, query + QUERY_OFFSET, 100 * RATIO * this_ratio, visited_list_pool_, start_points_, initial_seed_size, data, knn_graph,
                // visit_pq, l, r);
                // while (visit_pq.size() > KK){visit_pq.pop();}
                // while (visit_pq.size()) {
                //     nn_idx.emplace_back(visit_pq.top().second);
                //     visit_pq.pop();
                // }
                // reverse(nn_idx.begin(), nn_idx.end());
                // nn_idx.resize(KK);

                ///////////////// new method


                nns_searcher->SetEf(int(100 * RATIO * this_ratio));
                // nns_searcher->SetEf(qt2_ef);
                auto srched_res = new int[KK];
                nns_searcher->Search2((float*)(query + QUERY_OFFSET), KK, srched_res, data, l, r); //// range filter 

                for (unsigned i = 0; i < KK; ++i) {
                    nn_idx.emplace_back((unsigned)(srched_res[i]));
                }

                delete [] srched_res;
            }
        }

        return nn_idx;
    }
    else if (qt == 3)
    {
        auto catgory = (unsigned)(v);

        if (bbf) { /// true
            auto [lp, rp] = findRangePositions(data_pair, l, r) ;
            return BFSearchXX(data_pair, lp, rp, data, query, catgory);
        }
        else
        {
            auto its_size = id2size[catgory];
            if (its_size >= BUILD_GRAPH_THRES) {
                // auto & minigraph = id2graph[catgory];
                // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> visit_pq;
                // searchSingleBaseLayerMXX(0, query + QUERY_OFFSET, 100 * RATIO * this_ratio, visited_list_pool_, data, minigraph, id2list[catgory],
                // visit_pq, l, r);
                // while (visit_pq.size() > KK) {
                //     visit_pq.pop();
                // }
                // while (visit_pq.size()) {
                //     size_t real_id = id2list[catgory][visit_pq.top().second];
                //     nn_idx.emplace_back(real_id);
                //     visit_pq.pop();
                // }



                auto & mini_searcher = id2srcher[catgory];
                mini_searcher->SetEf(int(100 * RATIO * this_ratio ));
                // mini_searcher->SetEf((int)(100 * RATIO * this_ratio * std::pow(1.0 * its_size / knn_graph.size(), 0.1) * 1.2));

                
                // std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>> visit_pq;
                auto srched_res = new int[KK];
                // mini_searcher->Search3((float*)(query + QUERY_OFFSET), KK, srched_res, visit_pq, data, l, r, id2list[catgory]);
                mini_searcher->Search3((float*)(query + QUERY_OFFSET), KK, srched_res, data, l, r, id2list[catgory]);
                for (unsigned i = 0; i < KK; ++i) {
                    // nn_idx.emplace_back(id2list[catgory][(unsigned)(srched_res[i])]);
                    nn_idx.emplace_back((unsigned)(srched_res[i]));
                }
                delete [] srched_res;

            } else {
                auto & this_list = id2list[catgory];
                vector<NNItem> nnlist;

                for (size_t j = 0; j < its_size; ++j)
                {
                    // // std::cout << "C:" << node[0] << " T:" << node[1] << "\n";
                    // if (node[0] != v)
                    //     continue;
                    auto n_id = this_list[j]; 

                    auto node = (float *)(data + n_id * 416);
                    if (node[1] >= l && node[1] <= r) {
                        auto dst = basicL2(data + n_id * 416 + DATA_OFFSET, query + QUERY_OFFSET, data_dim);
                        // auto dst = fstdistfunc_(data + n_id * 416 + DATA_OFFSET, query + QUERY_OFFSET, dist_func_param_);

                        nnlist.emplace_back(n_id, dst);

                    }
                }
                std::sort(nnlist.begin(), nnlist.end());
                for (size_t iter = 0; iter < KK && iter < nnlist.size(); ++iter)
                {
                    nn_idx.emplace_back(nnlist[iter].idx);
                }
                if (nn_idx.size() < KK) {
                    // cout << "compos nbhoods:\n";
                    unsigned id{0};
                    while (nn_idx.size() < KK) {
                        nn_idx.emplace_back(id++);
                    }
                }
            }


        }
        return nn_idx;
    }
    else
    {
        return std::vector<unsigned>();
    }
}


void extreme_fast_gen_gt(std::unordered_map<unsigned, unsigned> &id2size,
mapList &id2list, mapGraph & id2graph, graph_t & knn_graph, char *data, size_t data_size, size_t data_dim, char *query, size_t query_size, 
    mapList & dis2list,  mapGraph & dis2graph,
    std::vector<std::pair<float, unsigned>> & data_pair,
    std::unique_ptr<pyanns::GraphSearcherBase> & nns_searcher,
    std::unordered_map<unsigned, std::unique_ptr<pyanns::GraphSearcherBase>> & id2srcher,
vector<vector<unsigned>> &gt_graph)
{

    // vector<float> ratio_list{10, 3, 2, 2, 1, 1, 1, 1, 1, 1}; ///first nouse
    // vector<float> ratio_list_lvl2{10, 12, 7, 5, 5, 5, 4, 3, 3, 3}; ///qt2 0.993X

    // vector<float> ratio_list{10, 3, 2, 2, 1.9, 1.5, 1.4, 1.2, 1, 1}; ///second para 0.997x
    // vector<float> ratio_list_lvl2{10, 15, 9, 7, 6, 6, 5, 5, 5, 4};////ver2

    // vector<float> ratio_list{10, 4, 3, 2, 1.9, 1.5, 1.4, 1.4, 1.4, 1.3}; ///paraV3 0.9988
    // vector<float> ratio_list_lvl2{10, 15, 9, 7, 6, 6, 5, 5, 5, 4};

    // vector<float> ratio_list{10, 4.8, 3.4, 3, 2.9, 2.5, 2.4, 2, 1.9, 1.9}; ///paraV4
    // vector<float> ratio_list_lvl2{10, 20, 15, 12, 9, 8, 7, 7, 6.6, 6}; ///0.999603



    // vector<float> ratio_list{10, 5.9, 4.9, 4, 3.5, 3.0, 3.0, 2.9, 2.5, 2.5}; ///paraV5
    // vector<float> ratio_list_lvl2{10, 20, 19, 15, 12, 11, 9, 8, 7.7, 7.3};

    // vector<float> ratio_list{10, 6.5, 5.4, 4.3, 3.8, 3.2, 3.0, 2.9, 2.5, 2.5}; ///paraV5.5 ---run0
    // vector<float> ratio_list{10, 6.7, 5.7, 4.5, 3.8, 3.2, 3.0, 2.9, 2.5, 2.5}; ///paraV5.5 ---run1
    vector<float> ratio_list{10, 7.2, 5.7, 4.5, 3.8, 3.3, 3.0, 3, 2.7, 2.7}; ///paraV5.5 ---run2
    
    vector<float> ratio_list_lvl2{10, 20, 19, 15, 12, 11, 9, 8, 7.7, 7.3};//


//////good this + 2300 2200 

    // vector<float> ratio_list{10, 8.4, 6, 5, 3.5, 4, 4, 4, 4, 4}; ///paraV6
    // vector<float> ratio_list_lvl2{10, 20, 19, 15, 12, 11, 9, 8, 7.7, 7.3};

    // vector<float> ratio_list{10, 8.8, 6.3, 5.3, 4, 4, 4, 4, 4, 4}; ///paraV6.5
    // // vector<float> ratio_list_lvl2{10, 20, 22, 18, 15, 13, 11, 9.5, 8.6, 8.2}; //// run3
    // // vector<float> ratio_list_lvl2{10, 20, 24, 19.5, 17.5, 15.3, 13.1, 10.5, 9.1, 8.5}; //// run4
    // // vector<float> ratio_list_lvl2{10, 20, 23, 18.5, 16.5, 15, 12.1, 10.5, 9.1, 9}; //// run5
    // vector<float> ratio_list_lvl2{10, 20, 22.5, 18.4, 15.5, 13.5, 11.5, 9.5, 8.6, 8.5}; //// run6

    // vector<float> ratio_list{10, 9, 6.8, 5.7, 4, 4, 4, 4, 4, 4}; ///paraV6.6
    // vector<float> ratio_list_lvl2{10, 20, 20, 17, 14, 12, 10, 9, 8.3, 8};

// vector<float> ratio_list{10, 8.9, 6.4, 5.5, 4.2, 4, 4, 4, 4, 4}; ///paraV6.7
// vector<float> ratio_list_lvl2{10, 20, 22.5, 18.4, 15.5, 13.5, 11.5, 9.5, 8.6, 8.5}; //// run1


    // vector<float> ratio_list{10, 8.7, 6.3, 5.5, 4, 5, 5, 5, 5, 5}; ///paraV7
    // vector<float> ratio_list_lvl2{10, 20, 14, 11, 8, 7, 7.5, 7.5, 7.6, 7};

    // vector<float> ratio_list{10, 8.7, 7, 7, 7, 7, 7, 7, 7, 7}; ///paraV7.1
    // vector<float> ratio_list_lvl2{10, 15, 14, 11, 8, 7, 7.5, 7.5, 7.6, 7};

    // vector<float> qt2ef_list{2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500};


    if (gt_graph.size())
    {
        gt_graph.clear();
    }
    gt_graph.resize(query_size);

    vector<vector<unsigned>> qry_groups(4);

    std::cout << "query_size = " << query_size << "\n";


    for (unsigned i = 0; i < query_size; ++i) {
            auto qq = (float *)(query + i * 416);
            float qt = qq[0];
            // float v = qq[1];
            // float l = qq[2];
            // float r = qq[3];
            qry_groups[unsigned(qt)].emplace_back(i);
    }
    // for (auto & group : qry_groups) {
    //     cout << "group_size = " << group.size() << "\n";
    // }

    // std::cout << "query_size = " << query_size << "\n";

// ///// search by groups 
for (unsigned gid = 0; gid < 4; ++gid) {

    auto & nodes = qry_groups[gid];

#pragma omp parallel for num_threads(32) schedule(dynamic, 1)
for (unsigned iter = 0; iter < nodes.size(); ++iter)
    {
        auto i = nodes[iter];
        gt_graph[i] = 
            extreme_fast_get_nbhoodLarge(id2size, id2list, id2graph, knn_graph, data, data_size, 
            data_dim, query + i * 416, 
            dis2list, dis2graph,
            ratio_list ,   ratio_list_lvl2,
            data_pair, nns_searcher, id2srcher
            );
    }
}
}




int main(int argc, char **argv)
{

    auto statt_max = std::chrono::high_resolution_clock::now();


    unsigned data_dim = 100;


    string source_path = "dummy-data.bin";
    string query_path = "dummy-queries.bin";
    string knn_save_path = "output.bin";

    // Also accept other path for source data
    if (argc > 1)
    {
        source_path = string(argv[1]);
    }
    if (argc > 2)
    {
        query_path = string(argv[2]);
    }
    if (argc > 3)
    {
        knn_save_path = string(argv[3]);
    }
    uint32_t num_data_dimensions = 102;


    size_t data_size;
    auto data = LoadBin(source_path, num_data_dimensions, data_size);
    std::cout << "data_size = " << data_size << "\n";


    auto pure_data = new float[data_size * data_dim];

#pragma omp parallel for 
    for (unsigned i = 0 ; i < data_size; ++i) {
        memcpy((char *)(pure_data + i * data_dim), (char *)(data + i * 416 + DATA_OFFSET), data_dim * sizeof(float));
    }



    std::vector<std::pair<float, unsigned>> data_pair(data_size);

#pragma omp parallel for 
    for (unsigned i = 0; i < data_size; ++i) {
        auto this_data = (float *) (data + i * 416);
        auto t = this_data[1];
        data_pair[i].first = t;
        data_pair[i].second = i;
    }


    std::sort(data_pair.begin(), data_pair.end());

    uint32_t num_query_dimensions = num_data_dimensions + 2;
    size_t query_size;
    auto query = LoadBin(query_path, num_query_dimensions, query_size);
    std::cout << "query_size = " << query_size << "\n";

    std::vector<unsigned> cat_ids;
    std::unordered_map<unsigned, unsigned> id2size;
    mapList id2list;
    mapGraph id2graph;
    std::unordered_map<unsigned, std::unique_ptr<pyanns::GraphSearcherBase>> id2srcher;

    cout << "data_size = " << data_size << "------\n";


    for (unsigned i = 0; i < data_size; ++i) {
        auto node = (float *)(data + (size_t)i * 416);
        auto c = unsigned(node[0]);
        if (id2list.find(c) != id2list.end()) {
            id2list[c].emplace_back(i);
        } else {
            cat_ids.emplace_back(c);
            id2list[c] = std::vector<unsigned>{i};
        }
    }

    cout << "------2\n";


    std::sort(cat_ids.begin(), cat_ids.end());

    cout << "cat_ids.size = " << cat_ids.size()
     << "------3.1\n";


    for (auto c : cat_ids){
        auto & list = id2list[c];
        id2size[c] = list.size(); 
        std::sort(list.begin(), list.end()); 
    }

    cout << "------3\n";


    rnndescent::Matrix<float> base_data;
    rnndescent::rnn_para rnn_idx_para;
    rnn_idx_para.S = 25;///25;///25;////36; // so far so good
    rnn_idx_para.T1 = 3;
    rnn_idx_para.T2 = 8;


    for (auto c : cat_ids) {
        // continue;
        auto & list = id2list[c]; 
        size_t small_size = id2size[c];
        if (small_size >= BUILD_GRAPH_THRES) {///TODO:


            float * small_vec = new float[small_size * data_dim]; 

#pragma omp parallel for
            for (unsigned tmpiter = 0; tmpiter < small_size; ++tmpiter) {
                auto id = list[tmpiter]; 
                memcpy((char *)(small_vec + tmpiter * data_dim), (float*)(data + id * 416 + DATA_OFFSET), data_dim * sizeof(float));
            }

            base_data.resize(small_size, data_dim);
            base_data.batch_add_test(small_vec, small_size);

            // unsigned tmpiter{0};
            // for (auto id : list) {
            //     base_data.add_test((float*)(data + id * 416 + DATA_OFFSET));
            //     memcpy((char *)(small_vec + tmpiter * data_dim), (float*)(data + id * 416 + DATA_OFFSET), data_dim * sizeof(float));
            //     ++tmpiter;
            // }

            // for (unsigned tmpiter = 0; tmpiter < small_size; ++tmpiter) {
            //     auto id = list[tmpiter]; 
            //     base_data.add_test((float*)(data + id * 416 + DATA_OFFSET));
            // }

            

            rnndescent::MatrixOracle<float, rnndescent::metric::l2sqr> oracle(base_data);
            
            std::unique_ptr<rnndescent::RNNDescent> index(new rnndescent::RNNDescent(oracle, rnn_idx_para));

            index->build(oracle.size(), false);

            id2graph[c] = std::vector<std::vector<unsigned>>();

            index->extract_index_graph(id2graph[c]);


            pyanns::Graph<int> mininns_graph;
            mininns_graph.load(id2graph[c], 32); /// max_edges

            std::unique_ptr<pyanns::GraphSearcherBase> mininns_searcher(pyanns::create_searcher(std::move(mininns_graph), "L2", "SQ8U"));

            mininns_searcher->SetData(small_vec, small_size, data_dim);
            mininns_searcher->Optimize();
            mininns_searcher->SetEf(1200);
            id2srcher[c]  = std::move(mininns_searcher);
            delete [] small_vec;
        }
    }

    graph_t knn_graph;


            base_data.resize(data_size, data_dim);
            base_data.batch_add_test(pure_data, data_size);
            // for (unsigned id = 0; id < data_size; ++id) {
            //     base_data.add_test((float*)(data + (size_t)id * 416 + DATA_OFFSET));
            // }


            rnndescent::MatrixOracle<float, rnndescent::metric::l2sqr> oracle(base_data);
            std::unique_ptr<rnndescent::RNNDescent> index(new rnndescent::RNNDescent(oracle, rnn_idx_para));

                    index->build(oracle.size(), true);

            index->extract_index_graph(knn_graph);


    ////// dis the time range :: out of date 
                                                    unsigned time_num_groups{NGPS};

                                                    std::vector<unsigned> dis_ids;
                                                    std::unordered_map<unsigned, unsigned> dis2size;
                                                    mapList dis2list;
                                                    mapGraph dis2graph;


    {
        auto  end = std::chrono::high_resolution_clock::now();
        auto time_costs = (1.0 * std::chrono::duration_cast<std::chrono::milliseconds>(end - statt_max).count() / 1000.0);
        cout << "@ INDEX time_costs = " << time_costs << "\n";
    }



///// for search 
    pyanns::Graph<int> nns_graph;
    nns_graph.load(knn_graph, 32);

    std::unique_ptr<pyanns::GraphSearcherBase> nns_searcher(pyanns::create_searcher(std::move(nns_graph), "L2", "SQ8U"));

    nns_searcher->SetData(pure_data, data_size, data_dim);
    nns_searcher->Optimize();
    nns_searcher->SetEf(1500);




    auto start = std::chrono::high_resolution_clock::now();

    vector<vector<unsigned>> gtttt;

    extreme_fast_gen_gt(id2size, id2list, id2graph, knn_graph, data, data_size,data_dim,
    query,query_size, 
    dis2list, dis2graph, 
    data_pair,
    nns_searcher,
        id2srcher,
     gtttt);


    auto  end = std::chrono::high_resolution_clock::now();

    auto time_costs = (1.0 * std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0);

    cout << "@ NNS time_costs = " << time_costs << "\n";

    SaveKNN(gtttt, knn_save_path);

    free(data);
    free(query);
    delete [] pure_data;
    return 0;
}