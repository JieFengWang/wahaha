#ifndef NNVERTEX_H
#define NNVERTEX_H

#include <iostream>
#include <math.h>
#include <limits>

using Idx_Type = unsigned;
using Dst_Type = float; 

using namespace std;

constexpr float EPS_h = 1e-6;


struct NNItem
{
    /* data */
    Idx_Type idx    = 0;
    Dst_Type dst    = std::numeric_limits<Dst_Type>::max();

    NNItem(void){}

    NNItem(Idx_Type idx1) : idx(idx1) {}
    NNItem(Idx_Type idx1, Dst_Type dst1) : idx(idx1), dst(dst1){}

    void setIdx(Idx_Type idx1){
        this->idx = idx1;
    }
    void setDst(Dst_Type dst1){
        this->dst = dst1;
    }

    friend std::ostream & operator<<(std::ostream & os, const NNItem & e){
        os << "NNItem: " 
           << "idx = "  << e.idx
           << " dst = " << e.dst << "\n";
        return os;
    }
    bool operator<(const NNItem& other) const {
        if (fabs(this->dst - other.dst) < EPS_h)
            return this->idx < other.idx;
        return this->dst < other.dst;
    }
    bool operator==(const NNItem& other) const {
        return this->idx == other.idx &&
            (fabs(this->dst - other.dst) < EPS_h);
    }
    bool operator>=(const NNItem& other) const { return !(*this < other); }
    bool operator<=(const NNItem& other) const {
        return (*this == other) || (*this < other);
    }
    bool operator>(const NNItem& other) const { return !(*this <= other); }
    bool operator!=(const NNItem& other) const { return !(*this == other); }
};


#endif