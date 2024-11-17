
#ifndef METRICS_HPP
#define METRICS_HPP

/**
 * @file metrics.hpp
 * 
 * @breif: implement distance measures
 * 
 * @author Jie-Feng Wang
 * 
*/


template<typename Dat_Type>
float basicL2(const Dat_Type *, const Dat_Type *, const unsigned int);


template<typename Dat_Type>
float basicL2(const Dat_Type * vec1, const Dat_Type * vec2, const unsigned int dim){
    float res = 0.0;
    // Dat_Type * v1 = (Dat_Type *) vec1;
    // Dat_Type * v2 = (Dat_Type *) vec2;
    float * v1 = (float *) vec1;
    float * v2 = (float *) vec2;
    for (unsigned int i = 0; i < dim; ++i){
        float diff = (float)(*v1 - *v2);
        res +=(float) (diff * diff);
        v1++;
        v2++;
    }
    return res;
}    
/*
this works under various data_types. 
BUT not supporting `char *` variables that are actually `float *`.
*/


/// ---- inner product

template<typename Dat_Type>
float basicInnerProduct(const Dat_Type * vec1, const Dat_Type * vec2, const unsigned int dim);


template<typename Dat_Type>
float basicInnerProduct(const Dat_Type * vec1, const Dat_Type * vec2, const unsigned int dim){
    float res = 0.0;
    Dat_Type * v1 = (Dat_Type *) vec1;
    Dat_Type * v2 = (Dat_Type *) vec2;
    for (unsigned int i = 0; i < dim; ++i){

        float product = (float)(*v1 * *v2);
        res += product;
        v1++;
        v2++;
    }
    // return -1.0 * res;
    return 1.0 - res;
} 

#endif