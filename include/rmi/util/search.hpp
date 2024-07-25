#pragma once

#include <algorithm>
#include <vector>
#include <cstdio>
#include <iostream>
#include <immintrin.h>
#include <avx512fintrin.h>
#include <cstdint>

/**
 * Functor for performing linear search.
 */
struct LinearSearch {
    /**
     * Performs linear search in the interval [first,last) to find the first element that is not less than @t value.
     * @tparam InputIt input iterator type
     * @tparam T type of searched value
     * @param first, last iterators defining the partially-ordered range to examine
     * @param pred iterator to the predicted position (ignored)
     * @param value value to compare the elements to
     * @return iterator to the first element that is not less than @p value
     */
    template<typename InputIt, typename T>
    InputIt operator()(InputIt first, InputIt last, InputIt /* pred */, const T &value) {
        InputIt runner = first;
        for (; runner != last; ++runner)
            if (*runner >= value) return runner;
        return last;
    }
};


/**
 * Functor for performing model-biased linear search.
 */
struct ModelBiasedLinearSearch {
    /**
     * Performs model-biased linear search either in the interval [first,pred) or [pred, last) to find the first element
     * that is not less than @t value.
     * @tparam InputIt input iterator type
     * @tparam T type of searched value
     * @param first, last iterators defining the partially-ordered range to examine
     * @param pred iterator to the predicted position
     * @param value value to compare the elements to
     * @return iterator to the first element that is not less than @p value
     */
    template<typename InputIt, typename T>
    InputIt operator()(InputIt first, InputIt last, InputIt pred, const T &value) {
        InputIt runner = pred;
        if (*runner < value) {
            for (; runner < last; ++runner) // search right side
                if (*runner >= value) return runner;
            return last;
        } else {
            for (; runner >= first; --runner)// search left side
                if (*runner < value) return ++runner;
            return first;
        }
    }
};


/**
 * Functor for performing binary search.
 */
struct BinarySearch {
    /**
     * Performs binary search in the interval [first,last) to find the first element that is not less than @t value.
     * @tparam InputIt input iterator type
     * @tparam T type of searched value
     * @param first, last iterators defining the partially-ordered range to examine
     * @param pred iterator to the predicted position (ignored)
     * @param value value to compare the elements to
     * @return iterator to the first element that is not less than @p value
     */
    template<typename InputIt, typename T>
    InputIt operator()(InputIt first, InputIt last, InputIt /* pred */, const T &value) {
        return std::lower_bound(first, last, value);
    }
};


/**
 * Functor for performing model-biased binary search.
 */
struct ModelBiasedBinarySearch {
    /**
     * Performs model-biased binary search either in the interval [first,pred) or [pred, last) to find the first element
     * that is not less than @t value.
     * @tparam InputIt input iterator type
     * @tparam T type of searched value
     * @param first, last iterators defining the partially-ordered range to examine
     * @param pred iterator to the predicted position
     * @param value value to compare the elements to
     * @return iterator to the first element that is not less than @p value
     */
    template<typename InputIt, typename T>
    InputIt operator()(InputIt first, InputIt last, InputIt pred, const T &value) {
        if (*pred < value) return std::lower_bound(pred, last, value); // search right side
        else return std::lower_bound(first, pred, value); // search left side
    }
};


/**
 * Functor for performing exponential search.
 */
struct ExponentialSearch {
    /**
     * Performs exponential search in the interval [first,last) to find the first element that is not less than @t
     * value.
     * @tparam InputIt input iterator type
     * @tparam T type of searched value
     * @param first, last iterators defining the partially-ordered range to examine
     * @param pred iterator to the predicted position (ignored)
     * @param value value to compare the elements to
     * @return iterator to the first element that is not less than @p value
     */
    template<typename InputIt, typename T>
    InputIt operator()(InputIt first, InputIt last, InputIt /* pred */, const T &value) {
        if (*first >= value) return first;
        std::size_t bound = 1;
        InputIt prev = first;
        InputIt curr = prev + bound;
        while (curr < last and *curr < value) {
            bound *= 2;
            prev = curr;
            curr += bound;
        }
        return std::lower_bound(prev, std::min(curr + 1, last), value);
    }
};


/**
 * Functor for performing model-biased exponential search.
 */
struct ModelBiasedExponentialSearch {
    /**
     * first 부터 pred 까지 혹은 pred 부터 last까지만 계산하는게 모델 베이스
     * Performs model-biased exponential search either in the interval [first,pred) or [pred, last) to find the first
     * element that is not less than @t value.
     * @tparam InputIt input iterator type
     * @tparam T type of searched value
     * @param first, last iterators defining the partially-ordered range to examine
     * @param pred iterator to the predicted position
     * @param value value to compare the elements to
     * @return iterator to the first element that is not less than @p value
     */
    template<typename InputIt, typename T>
    InputIt operator()(InputIt first, InputIt last, InputIt pred, const T &value) {
        if (*pred < value) { // search right side
            std::size_t bound = 1;
            InputIt prev = pred;
            InputIt curr = prev + bound;
            while (curr < last and *curr < value) {
                bound *= 2;
                prev = curr;
                curr += bound;
            }
            return std::lower_bound(prev, std::min(curr + 1, last), value);
        } else { // search left side
            std::size_t bound = 1;
            InputIt prev = pred;
            InputIt curr = prev - bound;
            while (curr > first and *curr >= value) {
                bound *= 2;
                prev = curr;
                curr -= bound;
            }
            return std::lower_bound(std::max(first, curr), prev, value);
        }
    }
};


/*-------------------------------------------------------------------------------------------*/

/**
 * Functor for performing linear search - SIMD.
 * this function is written by YeoJin Oh
 */
struct LinearSearch_SIMD {

    template<typename InputIt, typename T>
    InputIt operator()(InputIt first, InputIt last, InputIt /* pred */, const T &value) {
        const T* data = &(*first); // iterator를 배열로 넣음
        __m512i values = _mm512_set1_epi32(value);
        size_t n = std::distance(first, last);
        const size_t step = 8;
        for (size_t i = 0; i < n; i+=step){
            __m512i vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&data[i]));
            __mmask8 cmp_mask = _mm512_cmplt_epi64_mask(vec, values);
            if (cmp_mask != 0) {
                // Find the index of the first element satisfying the condition
                size_t index = __builtin_ctz(cmp_mask);
                return first + i + index;
            }
        }
        return last;
    }
};

/**
 * Functor for performing model-biased linear search.
 */
struct ModelBiasedLinearSearch_SIMD {

    template<typename InputIt, typename T>
    InputIt operator()(InputIt first, InputIt last, InputIt pred, const T &value) {
        InputIt runner = pred;
        //오른쪽 탐색
        if(*runner < value){
            const T* data = &(*runner); // iterator를 배열로 넣음
            __m512i values = _mm512_set1_epi32(value);
            size_t n = std::distance(runner, last);
            const size_t step = 8;
            for (size_t i = 0; i < n; i+=step){
                __m512i vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&data[i]));
                __mmask8 cmp_mask = _mm512_cmplt_epi64_mask(vec, values);
                if (cmp_mask != 0) {
                    // Find the index of the first element satisfying the condition
                    size_t index = __builtin_ctz(cmp_mask);
                    return first + i + index;
                }
            }
        return last;
        } else{
            const T* data = &(*first); // iterator를 배열로 넣음
            __m512i values = _mm512_set1_epi32(value);
            size_t n = std::distance(first, runner);
            const size_t step = 8;
            for (size_t i = 0; i < n; i+=step){
                __m512i vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&data[i]));
                __mmask8 cmp_mask = _mm512_cmplt_epi64_mask(vec, values);
                if (cmp_mask != 0) {
                    // Find the index of the first element satisfying the condition
                    size_t index = __builtin_ctz(cmp_mask);
                    return first + i + index;
                }
            }
            return last;
        }
    }
};

// bit scan reverse algorithm
//주어진 값 x에서 가장 오른쪽에 있는 비트의 인덱스 찾음.
// 입력값으로 해당 값의 가장 오른쪽에 있는 비트의 인덱스 반환 
inline intptr_t bsr(size_t x) {
    intptr_t ret = -1;
    while (x) {
        x >>= 1;
        ++ret;
    }
    return ret;
}

/**
 * Functor for performing binary search.
 */
intptr_t MINUS_ONE = -1;
struct BinarySearch_Branchless  {

    template<typename InputIt, typename T>
    InputIt operator()(InputIt first, InputIt last, InputIt /* pred */, const T &value) {
        size_t n = std::distance(first, last); // 전체 search 할 data의 크기 계산
        intptr_t pos = MINUS_ONE; // 
        intptr_t logstep = bsr(n);
        intptr_t step = intptr_t(1) << logstep;
        while(step > 0){
            pos = (*(first + step) < value ? pos + step : pos);
            step >>= 1;
        }
        return first + (pos + 1);
    }
};

struct ModelBiasedBinarySearch_Branchless {

    template<typename InputIt, typename T>
    InputIt operator()(InputIt first, InputIt last, InputIt pred, const T &value) {
        if (*pred < value) {
            size_t n = std::distance(pred, last); // 전체 search 할 data의 크기 계산
            intptr_t pos = MINUS_ONE; // 
            intptr_t logstep = bsr(n);
            intptr_t step = intptr_t(1) << logstep;
            while(step > 0){
                pos = (*(first + step) < value ? pos + step : pos);
                step >>= 1;
            }
            return first + (pos + 1);
        } 
        else{
            size_t n = std::distance(first, pred); // 전체 search 할 data의 크기 계산
            intptr_t pos = MINUS_ONE; // 
            intptr_t logstep = bsr(n);
            intptr_t step = intptr_t(1) << logstep;
            while(step > 0){
                pos = (*(first + step) < value ? pos + step : pos);
                step >>= 1;
            }
            return first + (pos + 1);
        }
    }
};