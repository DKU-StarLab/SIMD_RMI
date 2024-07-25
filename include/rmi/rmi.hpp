#pragma once
#include <chrono>
#include <algorithm>
#include <vector>
#include <immintrin.h>
#include <boost/align/aligned_allocator.hpp>
#include <iomanip> 
#include <cmath>

namespace rmi {

/**
 * Struct to hold the approximated position and error bounds returned by the index.
 */
struct Approx {
    std::size_t pos; ///< The estimated position of the key.
    std::size_t lo;  ///< The lower bound of the search range.
    std::size_t hi;  ///< The upper bound of the search range.
};

struct Approx_SIMD {
    std::vector<std::size_t> pos; ///< The estimated position of the key.
    std::vector<std::size_t> lo;  ///< The lower bound of the search range.
    std::vector<std::size_t> hi;  ///< The upper bound of the search range.
};



/**
 * This is a reimplementation of a two-layer recursive model index (RMI) supporting a variety of (monotonic) models.
 * RMIs were invented by Kraska et al. (https://dl.acm.org/doi/epdf/10.1145/3183713.3196909).
 *
 * Note that this is the base class which does not provide error bounds.
 *
 * @tparam Key the type of the keys to be indexed
 * @tparam Layer1 the type of the model used in layer1
 * @tparam Layer2 the type of the models used in layer2
 */
template<typename Key, typename Layer1, typename Layer2>
class Rmi
{
    using key_type = Key;
    using layer1_type = Layer1;
    using layer2_type = Layer2;
    // using AlignedVector_double = std::vector<double, boost::alignment::aligned_allocator<double, 64>>;
    // using AlignedVector_uint = std::vector<uint64_t, boost::alignment::aligned_allocator<uint64_t, 64>>;

    protected:
    std::size_t n_keys_;      ///< The number of keys the index was built on.
    std::size_t layer2_size_; ///< The number of models in layer2.
    layer1_type l1_;          ///< The layer1 model.
    layer2_type *l2_;         ///< The array of layer2 models.
    std::size_t switch_;     ///< The number of keys that switch SIMD/SISD
    // AlignedVector_double intercepts; // layer2 예측할때 model에서 함수호출하는것보다 vector로 미리저장하는것이 더 빠름//
    // AlignedVector_double slopes;     // layer2 예측할때 model에서 함수호출하는것보다 vector로 미리저장하는것이 더 빠름//
    // AlignedVector_uint seg_ele_counter;
    __m512i zero = _mm512_set1_epi64(0);
    __m512i n_keys_st = _mm512_set1_epi64((n_keys_ - 1));
    std::vector<std::size_t> seg_start_key;
    __m512i max_segment;
    __m512i min_segment;

    public:
    /**
     * Default constructor.
     */
    Rmi() = default;

    /**
     * Builds the index with @p layer2_size models in layer2 on the sorted @p keys.
     * @param keys vector of sorted keys to be indexed
     * @param layer2_size the number of models in layer2
     */
    Rmi(const std::vector<key_type> &keys, const std::size_t layer2_size, const std::size_t switch_n)
        : Rmi(keys.begin(), keys.end(), layer2_size, switch_n) { }


    /**
     * Builds the index with @p layer2_size models in layer2 on the sorted keys in the range [first, last).
     * @param first, last iterators that define the range of sorted keys to be indexed
     * @param layer2_size the number of models in layer2
     */
    template<typename RandomIt>
    Rmi(RandomIt first, RandomIt last, const std::size_t layer2_size, const std::size_t switch_n)
        : n_keys_(std::distance(first, last)),
        layer2_size_(layer2_size),
        switch_(switch_n)
    {
        int switch_bit = 1;
        max_segment = _mm512_set1_epi64((layer2_size_ - 1));
        min_segment = _mm512_set1_epi64(0);
        // Train layer1.
        l1_ = layer1_type(first, last, 0, static_cast<double>(layer2_size) / n_keys_); // train with compression

        // Train layer2.
        l2_ = new layer2_type[layer2_size];
        std::size_t segment_start = 0;
        std::size_t segment_id = 0;
        uint64_t i = 0;
        // Assign each key to its segment.
        for ( ; i != n_keys_; ++i) {
            auto pos = first + i;
            std::size_t pred_segment_id = get_segment_id(*pos);
            // If a key is assigned to a new segment, all models must be trained up to the new segment.
            if (pred_segment_id > segment_id) {
                if (static_cast<std::size_t>(std::distance(first + segment_start, pos)) < switch_){
                    new (&l2_[segment_id]) layer2_type(first + segment_start, pos, segment_start, switch_bit);
                }
                else{
                    new (&l2_[segment_id]) layer2_type(first + segment_start, pos, segment_start, switch_);  
                }
                for (std::size_t j = segment_id + 1; j < pred_segment_id; ++j) {
                    new (&l2_[j]) layer2_type(pos - 1, pos, i - 1, switch_bit); // train other models on last key in previous segment
                }
                segment_id = pred_segment_id;
                segment_start = i;
            }
        }
        // Train remaining models.
        new (&l2_[segment_id]) layer2_type(first + segment_start, last, segment_start, switch_bit);
        for (std::size_t j = segment_id + 1; j < layer2_size; ++j) {
            new (&l2_[j]) layer2_type(last - 1, last, n_keys_ - 1, switch_bit); // train remaining models on last key
        }
    }

    /**
     * Destructor.
     */
    ~Rmi() { delete[] l2_; }

    /**
     * Returns the id of the segment @p key belongs to.
     * @param key to get segment id for
     * @return segment id of the given key
     */
    std::size_t get_segment_id(const key_type key) const {
        return std::clamp<double>(l1_.predict(key), 0, layer2_size_ - 1);
    }

    __m512i get_segment_id_bySIMD(std::vector<uint64_t>::const_iterator it) const {
        __m512d segment_id_d = l1_.predict_bySIMD(it); //8개 key의 해당한 segment id를 구함(double형)//
        __m512d segment_id_round = _mm512_roundscale_pd(segment_id_d, _MM_FROUND_TO_ZERO);  // segment id를 반올림(double형)//
        __m512i segment_id = _mm512_cvtpd_epi64(segment_id_round);                       // segment id를 int형 변환 //
        __m512i n_keys_st = _mm512_set1_epi64((layer2_size_ - 1));  // segment_id_clamp//
        __m512i max_value = _mm512_min_epi64(segment_id, n_keys_st); // segment_id_clamp//
        __m512i min_value = _mm512_max_epi64(max_value, _mm512_set1_epi64(0)); // segment_id_clamp//
        return min_value; //제한[0, (layer2_size_ - 1)]을 적용한 segment id를 return//
    }

    __m512i get_segment_id_SIMD(__m512d & keys) const {
        return _mm512_max_epi64(_mm512_min_epi64(l1_.predict_bySIMD_(keys), max_segment), min_segment);
    }

    /**
     * Returns a position estimate and search bounds for a given key.
     * @param key to search for
     * @return position estimate and search bounds
     */
    Approx search(const key_type key) const {
        auto segment_id = get_segment_id(key);
        std::size_t pred = std::clamp<double>(l2_[segment_id].predict(key), 0, n_keys_ - 1);
        return {pred, 0, n_keys_};
    }

    Approx_SIMD search_bySIMD(std::vector<uint64_t>::const_iterator it) const {
        Approx_SIMD result;
        const std::size_t keys_size = 8;
        //std::array<key_type, 4> tempKeys = {default_value, default_value, default_value, default_value};
        __m512d key_d = _mm512_set_pd(
            static_cast<double>(*(it + 7)), static_cast<double>(*(it + 6)),
            static_cast<double>(*(it + 5)), static_cast<double>(*(it + 4)),
            static_cast<double>(*(it + 3)), static_cast<double>(*(it + 2)),
            static_cast<double>(*(it + 1)), static_cast<double>(*it)
        );
        result.pos.resize(keys_size); // pos등 error bound 제한을 준 결과값을 저장하는 공간 미리 할당//
        result.lo.resize(keys_size);
        result.hi.resize(keys_size);

        __m512i segment_ids_simd_ = get_segment_id_bySIMD(it); 
        // layer 2 에서 위치를 예측할때 필요한 slope, intercept를 가져옴//
        __m512d slope = _mm512_set_pd(l2_[segment_ids_simd_[7]].slope_, l2_[segment_ids_simd_[6]].slope_
                                    , l2_[segment_ids_simd_[5]].slope_, l2_[segment_ids_simd_[4]].slope_
                                    , l2_[segment_ids_simd_[3]].slope_, l2_[segment_ids_simd_[2]].slope_
                                    , l2_[segment_ids_simd_[1]].slope_, l2_[segment_ids_simd_[0]].slope_);
        __m512d intercept = _mm512_set_pd(l2_[segment_ids_simd_[7]].intercept_, l2_[segment_ids_simd_[6]].intercept_
                                        , l2_[segment_ids_simd_[5]].intercept_, l2_[segment_ids_simd_[4]].intercept_
                                        , l2_[segment_ids_simd_[3]].intercept_, l2_[segment_ids_simd_[2]].intercept_
                                        , l2_[segment_ids_simd_[1]].intercept_, l2_[segment_ids_simd_[0]].intercept_);
        //fmadd함수로 key_d는 slope랑 곱하고 intercept랑 더 함//
        __m512d pred = _mm512_roundscale_pd(_mm512_fmadd_pd(slope, key_d, intercept), _MM_FROUND_TO_ZERO);
        // pred를 key범위에서 뻐서나지 않게 하기 위해 일단 uint64형으로 전환//
        __m512i pred_i = _mm512_cvtpd_epi64(pred);
        // clamp함수의 simd 대응//
        __m512i min_values = _mm512_max_epi64(pred_i, zero);
        __m512i max_values = _mm512_min_epi64(n_keys_st, min_values);
        //lo, hi, pos를 저장후 return//
        _mm512_storeu_si512((__m512i*)result.pos.data(), max_values);
        _mm512_storeu_si512((__m512i*)result.lo.data(), zero);
        _mm512_storeu_si512((__m512i*)result.hi.data(), n_keys_st);
    
        return result;
    }

    /**
     * Returns the number of keys the index was built on.
     * @return the number of keys the index was built on
     */
    std::size_t n_keys() const { return n_keys_; }

    /**
     * Returns the number of models in layer2.
     * @return the number of models in layer2
     */
    std::size_t layer2_size() const { return layer2_size_; }

    /**
     * Returns the size of the index in bytes.
     * @return index size in bytes
     */
    std::size_t size_in_bytes() {
        return l1_.size_in_bytes() + layer2_size_ * l2_[0].size_in_bytes() + sizeof(n_keys_) + sizeof(layer2_size_);
    }

    std::size_t err_time() { return 0; }
};


/**
 * Recursive model index with global absolute bounds.
 */
template<typename Key, typename Layer1, typename Layer2>
class RmiGAbs : public Rmi<Key, Layer1, Layer2>
{
    using base_type = Rmi<Key, Layer1, Layer2>;
    using key_type = Key;
    using layer1_type = Layer1;
    using layer2_type = Layer2;

    protected:
    std::size_t error_; ///< The error bound of the layer2 models.
    std::vector<uint64_t> seg_id_list;
    std::chrono::time_point<std::chrono::steady_clock> err_start;
    std::chrono::time_point<std::chrono::steady_clock> err_stop; 

    public:
    /**
     * Default constructor.
     */
    RmiGAbs() = default;

    /**
     * Builds the index with @p layer2_size models in layer2 on the sorted @p keys.
     * @param keys vector of sorted keys to be indexed
     * @param layer2_size the number of models in layer2
     */
    RmiGAbs(const std::vector<key_type> &keys, const std::size_t layer2_size, const std::size_t switch_n)
        : RmiGAbs(keys.begin(), keys.end(), layer2_size, switch_n) { }

    /**
     * Builds the index with @p layer2_size models in layer2 on the sorted keys in the range [first, last).
     * @param first, last iterators that define the range of sorted keys to be indexed
     * @param layer2_size the number of models in layer2
     */
    template<typename RandomIt>
    RmiGAbs(RandomIt first, RandomIt last, const std::size_t layer2_size, const std::size_t switch_n) : base_type(first, last, layer2_size, switch_n) {
        // Compute global absolute errror bounds.
        err_start = std::chrono::steady_clock::now();
        error_ = 0;

        // Compute Global absolute error bounds by SIMD
        for (std::size_t i = 0; i < base_type::n_keys_; i+=8) {
            __m512d key_d = _mm512_set_pd(
                                        static_cast<double>(*(first + i + 7)), static_cast<double>(*(first + i + 6)),
                                        static_cast<double>(*(first + i + 5)), static_cast<double>(*(first + i + 4)),
                                        static_cast<double>(*(first + i + 3)), static_cast<double>(*(first + i + 2)),
                                        static_cast<double>(*(first + i + 1)), static_cast<double>(*(first + i))); 
            __m512d pos = _mm512_set_pd(i+7, i+6, i+5, i+4, i+3, i+2, i+1, i);
            __m512i segment_ids_simd_ = base_type::get_segment_id_bySIMD(first + i); // get 8 keys segment_id
            __m512d slope = _mm512_set_pd(base_type::l2_[segment_ids_simd_[7]].slope_, base_type::l2_[segment_ids_simd_[6]].slope_
                                    , base_type::l2_[segment_ids_simd_[5]].slope_, base_type::l2_[segment_ids_simd_[4]].slope_
                                    , base_type::l2_[segment_ids_simd_[3]].slope_, base_type::l2_[segment_ids_simd_[2]].slope_
                                    , base_type::l2_[segment_ids_simd_[1]].slope_, base_type::l2_[segment_ids_simd_[0]].slope_);
            __m512d intercept = _mm512_set_pd(base_type::l2_[segment_ids_simd_[7]].intercept_, base_type::l2_[segment_ids_simd_[6]].intercept_
                                        , base_type::l2_[segment_ids_simd_[5]].intercept_, base_type::l2_[segment_ids_simd_[4]].intercept_
                                        , base_type::l2_[segment_ids_simd_[3]].intercept_, base_type::l2_[segment_ids_simd_[2]].intercept_
                                        , base_type::l2_[segment_ids_simd_[1]].intercept_, base_type::l2_[segment_ids_simd_[0]].intercept_);
            __m512d pred = _mm512_roundscale_pd(_mm512_fmadd_pd(slope, key_d, intercept), _MM_FROUND_TO_ZERO);
            uint64_t batch_abs_err = _mm512_reduce_max_epi64(_mm512_cvtpd_epi64(_mm512_abs_pd(_mm512_roundscale_pd(_mm512_sub_pd(pred, pos), _MM_FROUND_TO_ZERO))));
            error_ = batch_abs_err > error_ ? batch_abs_err : error_;
        }
        err_stop = std::chrono::steady_clock::now();
    }

    /**
     * Returns a position estimate and search bounds for a given key.
     * @param key to search for
     * @return position estimate and search bounds
     */
    Approx search(const key_type key) const {
        auto segment_id = base_type::get_segment_id(key);
        std::size_t pred = std::clamp<double>(base_type::l2_[segment_id].predict(key), 0, base_type::n_keys_ - 1);
        std::size_t lo = pred > error_ ? pred - error_ : 0;
        std::size_t hi = std::min(pred + error_ + 1, base_type::n_keys_);
        return {pred, lo, hi};
    }

    Approx_SIMD search_bySIMD(std::vector<uint64_t>::const_iterator it) const {
        Approx_SIMD result;
        const std::size_t keys_size = 8;
        //std::array<key_type, 4> tempKeys = {default_value, default_value, default_value, default_value};
        __m512d key_d = _mm512_set_pd(
            static_cast<double>(*(it + 7)), static_cast<double>(*(it + 6)),
            static_cast<double>(*(it + 5)), static_cast<double>(*(it + 4)),
            static_cast<double>(*(it + 3)), static_cast<double>(*(it + 2)),
            static_cast<double>(*(it + 1)), static_cast<double>(*it)
        );
        result.pos.resize(keys_size); // pos등 error bound 제한을 준 결과값을 저장하는 공간 미리 할당//
        result.lo.resize(keys_size);
        result.hi.resize(keys_size);

        //result.err.resize(keys_size);

        __m512i segment_ids_simd_ = base_type::get_segment_id_bySIMD(it);

        // layer 2 에서 위치를 예측할때 필요한 slope, intercept를 가져옴//
        __m512d slope = _mm512_set_pd(base_type::l2_[segment_ids_simd_[7]].slope_, base_type::l2_[segment_ids_simd_[6]].slope_
                                    , base_type::l2_[segment_ids_simd_[5]].slope_, base_type::l2_[segment_ids_simd_[4]].slope_
                                    , base_type::l2_[segment_ids_simd_[3]].slope_, base_type::l2_[segment_ids_simd_[2]].slope_
                                    , base_type::l2_[segment_ids_simd_[1]].slope_, base_type::l2_[segment_ids_simd_[0]].slope_);
        __m512d intercept = _mm512_set_pd(base_type::l2_[segment_ids_simd_[7]].intercept_, base_type::l2_[segment_ids_simd_[6]].intercept_
                                        , base_type::l2_[segment_ids_simd_[5]].intercept_, base_type::l2_[segment_ids_simd_[4]].intercept_
                                        , base_type::l2_[segment_ids_simd_[3]].intercept_, base_type::l2_[segment_ids_simd_[2]].intercept_
                                        , base_type::l2_[segment_ids_simd_[1]].intercept_, base_type::l2_[segment_ids_simd_[0]].intercept_);
        // key를 double형 변경//

        //fmadd함수로 key_d는 slope랑 곱하고 intercept랑 더 함//
        //__m512d pred = _mm512_sub_pd(_mm512_fmadd_pd(slope, key_d, intercept), _mm512_set1_pd(0.5));
        __m512d pred = _mm512_roundscale_pd(_mm512_fmadd_pd(slope, key_d, intercept), _MM_FROUND_TO_ZERO);
        // pos(pred)를 uint64 simd변환 & 제한을 적용후 preds에 저장 //
        __m512i pred_i = _mm512_cvtpd_epi64(pred);
        __m512i min_values = _mm512_max_epi64(pred_i, base_type::zero);
        __m512i max_values = _mm512_min_epi64(base_type::n_keys_st, min_values);
        // error bound를 적용해서 pos가 0 인 특수상황을 처리하기 위한 과정 (즉, pos가 0이면, lo=0, 아니면 lo = (pred_i - error)//
        __m512i error_simd = _mm512_set1_epi64(error_);
        __mmask8 compare_ = _mm512_cmpgt_epi64_mask(max_values, error_simd);
        __m512i lower_ = _mm512_sub_epi64(max_values, error_simd);
        __m512i lo_ = _mm512_mask_blend_epi64(compare_, base_type::zero, lower_);
        // error bound를 적용해서 pos+error가 n_keys_st보다 큰 특수상황을 처리하기 위한 과정 (즉, hi=n_keys_st, 아니면 hi = (pred_i + error)//
        __m512i upper_first = _mm512_add_epi64(max_values, error_simd);
        __m512i upper_second = _mm512_add_epi64(upper_first, _mm512_set1_epi64(1));
        __m512i hi_ = _mm512_min_epi64(upper_second, base_type::n_keys_st);

        //std::fill(result.err.begin(), result.err.end(), error_);

        _mm512_storeu_si512((__m512i*)result.pos.data(), max_values);
        _mm512_storeu_si512((__m512i*)result.lo.data(), lo_);
        _mm512_storeu_si512((__m512i*)result.hi.data(), hi_);

        return result;
    }
    /**
     * Returns the size of the index in bytes.
     * @return index size in bytes
     */
    std::size_t size_in_bytes() { return base_type::size_in_bytes() + sizeof(error_); }
    std::size_t err_time() { return std::chrono::duration_cast<std::chrono::nanoseconds>(err_stop - err_start).count(); }
};


/**
 * Recursive model index with global individual bounds.
 */
template<typename Key, typename Layer1, typename Layer2>
class RmiGInd : public Rmi<Key, Layer1, Layer2>
{
    using base_type = Rmi<Key, Layer1, Layer2>;
    using key_type = Key;
    using layer1_type = Layer1;
    using layer2_type = Layer2;

    protected:
    std::size_t error_lo_; ///< The lower error bound of the layer2 models.
    std::size_t error_hi_; ///< The upper error bound of the layer2 models.
    std::chrono::time_point<std::chrono::steady_clock> err_start;
    std::chrono::time_point<std::chrono::steady_clock> err_stop; 


    public:
    /**
     * Default constructor.
     */
    RmiGInd() = default;

    /**
     * Builds the index with @p layer2_size models in layer2 on the sorted @p keys.
     * @param keys vector of sorted keys to be indexed
     * @param layer2_size the number of models in layer2
     */
    RmiGInd(const std::vector<key_type> &keys, const std::size_t layer2_size, const std::size_t switch_n)
        : RmiGInd(keys.begin(), keys.end(), layer2_size, switch_n) { }

    /**
     * Builds the index with @p layer2_size models in layer2 on the sorted keys in the range [first, last).
     * @param first, last iterators that define the range of sorted keys to be indexed
     * @param layer2_size the number of models in layer2
     */
    template<typename RandomIt>
    RmiGInd(RandomIt first, RandomIt last, const std::size_t layer2_size, const std::size_t switch_n) : base_type(first, last, layer2_size, switch_n) {
        err_start = std::chrono::steady_clock::now();
        // Compute global absolute errror bounds.
        error_lo_ = 0;
        error_hi_ = 0;
        for (std::size_t i = 0; i != base_type::n_keys_; ++i) {
            key_type key = *(first + i);
            std::size_t segment_id = base_type::get_segment_id(key);
            std::size_t pred = std::clamp<double>(base_type::l2_[segment_id].predict(key), 0, base_type::n_keys_ - 1);
            if (pred > i) { // overestimation
                error_lo_ = std::max(error_lo_, pred - i);
            } else { // underestimation
                error_hi_ = std::max(error_hi_, i - pred);
            }
        }
        err_stop = std::chrono::steady_clock::now();
    }

    /**
     * Returns a position estimate and search bounds for a given key.
     * @param key to search for
     * @return position estimate and search bounds
     */
    Approx search(const key_type key) const {
        auto segment_id = base_type::get_segment_id(key);
        std::size_t pred = std::clamp<double>(base_type::l2_[segment_id].predict(key), 0, base_type::n_keys_ - 1);
        std::size_t lo = pred > error_lo_ ? pred - error_lo_ : 0;
        std::size_t hi = std::min(pred + error_hi_ + 1, base_type::n_keys_);
        return {pred, lo, hi};
    }

    Approx_SIMD search_bySIMD(std::vector<uint64_t>::const_iterator it) const {
        Approx_SIMD result;
        const std::size_t keys_size = 8;
        //std::array<key_type, 4> tempKeys = {default_value, default_value, default_value, default_value};
        __m512d key_d = _mm512_set_pd(
            static_cast<double>(*(it + 7)), static_cast<double>(*(it + 6)),
            static_cast<double>(*(it + 5)), static_cast<double>(*(it + 4)),
            static_cast<double>(*(it + 3)), static_cast<double>(*(it + 2)),
            static_cast<double>(*(it + 1)), static_cast<double>(*it)
        );
        result.pos.resize(keys_size); // pos등 error bound 제한을 준 결과값을 저장하는 공간 미리 할당//
        result.lo.resize(keys_size);
        result.hi.resize(keys_size);

        __m512i segment_ids_simd_ = base_type::get_segment_id_bySIMD(it);
        // layer 2 에서 위치를 예측할때 필요한 slope, intercept를 가져옴//
        __m512d slope = _mm512_set_pd(base_type::l2_[segment_ids_simd_[7]].slope_, base_type::l2_[segment_ids_simd_[6]].slope_
                                    , base_type::l2_[segment_ids_simd_[5]].slope_, base_type::l2_[segment_ids_simd_[4]].slope_
                                    , base_type::l2_[segment_ids_simd_[3]].slope_, base_type::l2_[segment_ids_simd_[2]].slope_
                                    , base_type::l2_[segment_ids_simd_[1]].slope_, base_type::l2_[segment_ids_simd_[0]].slope_);
        __m512d intercept = _mm512_set_pd(base_type::l2_[segment_ids_simd_[7]].intercept_, base_type::l2_[segment_ids_simd_[6]].intercept_
                                        , base_type::l2_[segment_ids_simd_[5]].intercept_, base_type::l2_[segment_ids_simd_[4]].intercept_
                                        , base_type::l2_[segment_ids_simd_[3]].intercept_, base_type::l2_[segment_ids_simd_[2]].intercept_
                                        , base_type::l2_[segment_ids_simd_[1]].intercept_, base_type::l2_[segment_ids_simd_[0]].intercept_);

        //fmadd함수로 key_d는 slope랑 곱하고 intercept랑 더 함//
        __m512d pred = _mm512_roundscale_pd(_mm512_fmadd_pd(slope, key_d, intercept), _MM_FROUND_TO_ZERO);
        // pos(pred)를 uint64 simd변환 & 제한을 적용후 preds에 저장 //
        __m512i pred_i = _mm512_cvtpd_epi64(pred);
        __m512i min_values = _mm512_max_epi64(pred_i, base_type::zero);
        __m512i max_values = _mm512_min_epi64(base_type::n_keys_st, min_values);
        // error bound를 적용해서 pos가 0 인 특수상황을 처리하기 위한 과정 (즉, pos가 0이면, lo=0, 아니면 lo = (pred_i - error_lo)//
        __m512i error_lo_simd = _mm512_set1_epi64(error_lo_);
        __m512i lower_ = _mm512_sub_epi64(max_values, error_lo_simd);
        __mmask8 compare_ = _mm512_cmpgt_epi64_mask(max_values, error_lo_simd);
        __m512i lo_ = _mm512_mask_blend_epi64(compare_, base_type::zero, lower_);
        // error bound를 적용해서 pos+error_hi가 n_keys_st보다 큰 특수상황을 처리하기 위한 과정 (즉, hi=n_keys_st, 아니면 hi = (pred_i + error_hi)//
        __m512i error_hi_simd = _mm512_set1_epi64(error_hi_);
        __m512i upper_first = _mm512_add_epi64(max_values, error_hi_simd);
        __m512i upper_second = _mm512_add_epi64(upper_first, _mm512_set1_epi64(1));
        __m512i hi_ = _mm512_min_epi64(upper_second, base_type::n_keys_st);

        _mm512_storeu_si512((__m512i*)result.pos.data(), max_values);
        _mm512_storeu_si512((__m512i*)result.lo.data(), lo_);
        _mm512_storeu_si512((__m512i*)result.hi.data(), hi_);

        return result;
    }

    /**
     * Returns the size of the index in bytes.
     * @return index size in bytes
     */
    std::size_t size_in_bytes() { return base_type::size_in_bytes() + sizeof(error_lo_) + sizeof(error_hi_); }
    std::size_t err_time() { return std::chrono::duration_cast<std::chrono::nanoseconds>(err_stop - err_start).count(); }
};


/**
 * Recursive model index with local absolute bounds.
 */
template<typename Key, typename Layer1, typename Layer2>
class RmiLAbs : public Rmi<Key, Layer1, Layer2>
{
    using base_type = Rmi<Key, Layer1, Layer2>;
    using key_type = Key;
    using layer1_type = Layer1;
    using layer2_type = Layer2;

    protected:
    std::vector<std::size_t> errors_; ///< The error bounds of the layer2 models.
    __m512i errors_simd = _mm512_set1_epi64(0);
    std::vector<std::size_t> combined_errors;
    std::chrono::time_point<std::chrono::steady_clock> err_start;
    std::chrono::time_point<std::chrono::steady_clock> err_stop; 

    public:
    /**
     * Default constructor.
     */
    RmiLAbs() = default;

    /**
     * Builds the index with @p layer2_size models in layer2 on the sorted @p keys.
     * @param keys vector of sorted keys to be indexed
     * @param layer2_size the number of models in layer2
     */
    RmiLAbs(const std::vector<key_type> &keys, const std::size_t layer2_size, const std::size_t switch_n)
        : RmiLAbs(keys.begin(), keys.end(), layer2_size, switch_n) { }

    /**
     * Builds the index with @p layer2_size models in layer2 on the sorted keys in the range [first, last).
     * @param first, last iterators that define the range of sorted keys to be indexed
     * @param layer2_size the number of models in layer2
     */
    template<typename RandomIt>
    RmiLAbs(RandomIt first, RandomIt last, const std::size_t layer2_size, const std::size_t switch_n) : base_type(first, last, layer2_size, switch_n) {
        // //初始化误差向量
        // uint64_t next_segment_count = 0;
        // // 处理剩余的元素
        // std::vector<double> combined_keys(8); // 当前段剩余的key和下一个段的部分key
        // std::vector<std::size_t> combined_positions(8); // 相应的位置
        // std::vector<double> combined_slopes(8);
        // std::vector<double> combined_intercepts(8);
        // std::vector<std::size_t> combined_segments(8); 
        // uint64_t combined_count = 0;

        // uint64_t k = 0;  // 当前偏移和索引位置
        // std::size_t i = 0;  // 初始段ID
        // while (i < base_type::layer2_size_) {
        //     std::size_t segment_id = i;
        //     double slope = base_type::slopes[segment_id];
        //     double intercept = base_type::intercepts[segment_id];
        //     __m512d slope_ = _mm512_set1_pd(slope);
        //     __m512d intercept_ = _mm512_set1_pd(intercept);

        //     uint64_t segment_size = base_type::seg_ele_counter[i] - next_segment_count;  // 本段内的key数量
        //     uint64_t full_simd_iters = segment_size / 8;            // SIMD计算的次数
        //     uint64_t remaining_elements = segment_size % 8;         // 剩余的key数量

        //     remaining_elements != 0 ? i : ++i;
        //     if (full_simd_iters == 0 && remaining_elements == 0) {
        //         i = i+1;
        //         continue;
        //     }

        //     std::cout << i << std::endl;

        //     // 处理完整的SIMD宽度块
        //     for (uint64_t j = 0; j < full_simd_iters; ++j) {
        //         __m512d key_d = _mm512_set_pd(
        //                             static_cast<double>(*(first + k + 7)), static_cast<double>(*(first + k + 6)),
        //                             static_cast<double>(*(first + k + 5)), static_cast<double>(*(first + k + 4)),
        //                             static_cast<double>(*(first + k + 3)), static_cast<double>(*(first + k + 2)),
        //                             static_cast<double>(*(first + k + 1)), static_cast<double>(*(first + k)));
        //         __m512i pos = _mm512_set_epi64((k+7), (k+6), (k+5), (k+4),
        //                                         (k+3), (k+2), (k+1), (k));
        //         __m512i pred = _mm512_cvtpd_epi64(_mm512_roundscale_pd(_mm512_fmadd_pd(slope_, key_d, intercept_), _MM_FROUND_TO_ZERO));
        //         __m512i error = _mm512_abs_epi64(_mm512_sub_epi64(pred, pos));
        //         uint64_t maxErr_ = _mm512_reduce_max_epi64(error);
        //         errors_[segment_id] = std::max(errors_[segment_id], maxErr_);
        //         k += 8;
        //     }

        //     if (remaining_elements != 0) {


        //         combined_count = 0;

        //         // 添加当前段中的剩余元素 해당segment에서 나머지 key를 load, 그외 해당key의 segment id, pos, slope, intercept, 해당segment의err도 load
        //         for (uint64_t j = 0; j < remaining_elements; ++j) {
        //             combined_keys[combined_count] = static_cast<double>(*(first + k + j));
        //             combined_positions[combined_count] = k + j;
        //             combined_slopes[combined_count] = slope;
        //             combined_intercepts[combined_count] = intercept;
        //             combined_segments[combined_count] = segment_id;
        //             combined_errors[combined_count] = errors_[segment_id];
        //             combined_count++;
        //         }   
        //         k += remaining_elements;

        //         // 添加后续段中的元素，直到凑满8个键 다음segment에서 자리를 채워줌
        //         while (combined_count < 8 && i + 1 < base_type::layer2_size_) {
        //             ++i;
        //             std::size_t next_segment_id = i;
        //             uint64_t next_segment_size = base_type::seg_ele_counter[next_segment_id];
        //             next_segment_count = std::min(next_segment_size, 8 - combined_count); //다음segment내의 원소가 나머지 자리보다 적을때 상황을 대응
        //             double next_seg_slope = base_type::slopes[next_segment_id];
        //             double next_seg_intercept = base_type::intercepts[next_segment_id];

        //             for (uint64_t j = 0; j < next_segment_count; ++j) {
        //                 combined_keys[combined_count] = static_cast<double>(*(first + k + j));
        //                 combined_positions[combined_count] = k + j;
        //                 combined_slopes[combined_count] = next_seg_slope;
        //                 combined_intercepts[combined_count] = next_seg_intercept;
        //                 combined_segments[combined_count] = next_segment_id;
        //                 combined_errors[combined_count] = errors_[next_segment_id];
        //                 combined_count++;
        //             }
        //             k += next_segment_count;

        //             if (combined_count == 8) {
        //                 // 执行SIMD计算
        //                 __m512d slope = _mm512_loadu_pd(combined_slopes.data());
        //                 __m512d intercept = _mm512_loadu_pd(combined_intercepts.data());
        //                 __m512i errors_simd = _mm512_loadu_si512(combined_errors.data());
        //                 __m512d key_d = _mm512_loadu_pd(combined_keys.data());
        //                 __m512i pos = _mm512_loadu_si512(combined_positions.data());
        //                 __m512i pred = _mm512_cvtpd_epi64(_mm512_roundscale_pd(_mm512_fmadd_pd(slope, key_d, intercept), _MM_FROUND_TO_ZERO));
        //                 __m512i maxErrs = _mm512_max_epi64(_mm512_abs_epi64(_mm512_sub_epi64(pred, pos)), errors_simd);

        //                 // 将 maxErrs 存储到临时数组
        //                 std::size_t temp_errors[8];
        //                 _mm512_storeu_si512(temp_errors, maxErrs);

        //                 // 更新 errors_ 向量
        //                 for (int j = 0; j < 8; ++j) {
        //                     errors_[combined_segments[j]] = std::max(errors_[combined_segments[j]], temp_errors[j]);
        //                 }

        //                 combined_count = 0;  // 重置组合计数器
        //                 break;
        //             }
        //         }
        //     }
        //     else{
        //         next_segment_count = 0;
        //     }

        //     // 如果剩余元素不足8个，使用SISD计算
        //     if (combined_count > 0) {
        //         for (uint64_t j = 0; j < combined_count; ++j) {
        //             key_type key = static_cast<key_type>(combined_keys[j]);
        //             std::size_t pred = std::clamp<double>(base_type::l2_[combined_segments[j]].predict(key), 0, base_type::n_keys_ - 1);
        //             std::size_t pos = static_cast<std::size_t>(combined_positions[j]);
        //             if (pred > pos) { // 过高估计
        //                 errors_[combined_segments[j]] = std::max(errors_[combined_segments[j]], pred - pos);
        //             } else { // 过低估计
        //                 errors_[combined_segments[j]] = std::max(errors_[combined_segments[j]], pos - pred);
        //             }
        //         }
        //     }

        // }
        // err_stop = std::chrono::steady_clock::now();


        err_start = std::chrono::steady_clock::now();
        errors_ = std::vector<std::size_t>(layer2_size, 0);
        combined_errors = std::vector<std::size_t>(8, 0);

        __m512d keys = _mm512_set1_pd(0);
        __m512d slopes = _mm512_set1_pd(0);
        __m512d intercepts = _mm512_set1_pd(0);

        __m512i add8 = _mm512_set1_epi64(8);
        __m512i preds = _mm512_set1_epi64(0);
        __m512i pos = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);

        __m512i segments =  _mm512_set1_epi64(0);
        __m512i new_errors = _mm512_set1_epi64(0);

        // Simple SIMD
        for (std::size_t i = 0; i < base_type::n_keys_; i+=8) {
            keys = _mm512_set_pd(static_cast<double>(*(first + i + 7)), static_cast<double>(*(first + i + 6)),
                                static_cast<double>(*(first + i + 5)), static_cast<double>(*(first + i + 4)),
                                static_cast<double>(*(first + i + 3)), static_cast<double>(*(first + i + 2)),
                                static_cast<double>(*(first + i + 1)), static_cast<double>(*(first + i)));

            segments = base_type::get_segment_id_SIMD(keys); // get 8 keys segment_id

            slopes = _mm512_set_pd(base_type::l2_[segments[7]].slope_, base_type::l2_[segments[6]].slope_
                                    ,base_type::l2_[segments[5]].slope_, base_type::l2_[segments[4]].slope_
                                    ,base_type::l2_[segments[3]].slope_, base_type::l2_[segments[2]].slope_
                                    ,base_type::l2_[segments[1]].slope_, base_type::l2_[segments[0]].slope_);

            intercepts = _mm512_set_pd(base_type::l2_[segments[7]].intercept_,base_type::l2_[segments[6]].intercept_
                                        ,base_type::l2_[segments[5]].intercept_,base_type::l2_[segments[4]].intercept_
                                        ,base_type::l2_[segments[3]].intercept_, base_type::l2_[segments[2]].intercept_
                                        ,base_type::l2_[segments[1]].intercept_, base_type::l2_[segments[0]].intercept_);

            preds = _mm512_cvtpd_epi64(_mm512_roundscale_pd(_mm512_fmadd_pd(slopes, keys, intercepts), _MM_FROUND_TO_ZERO));

            __mmask8 mask = _mm512_cmpeq_epi64_mask(segments, _mm512_set1_epi64(segments[0]));
            bool all_equal = (mask == 0xFF);
            if (all_equal) {
                uint64_t maxErr_ = _mm512_reduce_max_epi64(_mm512_abs_epi64(_mm512_sub_epi64(preds, pos)));
                errors_[segments[0]] = std::max(errors_[segments[0]], maxErr_);
            }

            else {
                new_errors = _mm512_set_epi64(errors_[segments[7]], errors_[segments[6]],
                                            errors_[segments[5]], errors_[segments[4]],
                                            errors_[segments[3]], errors_[segments[2]],
                                            errors_[segments[1]], errors_[segments[0]]);
                new_errors = _mm512_max_epi64(_mm512_abs_epi64(_mm512_sub_epi64(preds, pos)), new_errors);

            // 동시에 동일한 값으르 접근하면 안 됨.
            // 논문에 해당코드 사용 불가 이슈 작승
            //_mm512_i64scatter_epi64(static_cast<void*>(errors_.data()), segments, new_errors, 8);
                errors_[segments[0]] = new_errors[0] > errors_[segments[0]] ? new_errors[0] : errors_[segments[0]]; 
                errors_[segments[1]] = new_errors[1] > errors_[segments[1]] ? new_errors[1] : errors_[segments[1]];
                errors_[segments[2]] = new_errors[2] > errors_[segments[2]] ? new_errors[2] : errors_[segments[2]]; 
                errors_[segments[3]] = new_errors[3] > errors_[segments[3]] ? new_errors[3] : errors_[segments[3]];
                errors_[segments[4]] = new_errors[4] > errors_[segments[4]] ? new_errors[4] : errors_[segments[4]];
                errors_[segments[5]] = new_errors[5] > errors_[segments[5]] ? new_errors[5] : errors_[segments[5]];
                errors_[segments[6]] = new_errors[6] > errors_[segments[6]] ? new_errors[6] : errors_[segments[6]];
                errors_[segments[7]] = new_errors[7] > errors_[segments[7]] ? new_errors[7] : errors_[segments[7]];
            }
            pos =_mm512_add_epi64(pos, add8);
        }
        err_stop = std::chrono::steady_clock::now();

    }

    /**
     * Returns a position estimate and search bounds for a given key.
     * @param key to search for
     * @return position estimate and search bounds
     */
    Approx search(const key_type key) const {
        auto segment_id = base_type::get_segment_id(key);
        std::size_t pred = std::clamp<double>(base_type::l2_[segment_id].predict(key), 0, base_type::n_keys_ - 1);
        std::size_t err = errors_[segment_id];
        std::size_t lo = pred > err ? pred - err : 0;
        std::size_t hi = std::min(pred + err + 1, base_type::n_keys_);
        return {pred, lo, hi};
    }

    Approx_SIMD search_bySIMD(std::vector<uint64_t>::const_iterator it) const {
        Approx_SIMD result;
        const std::size_t keys_size = 8;
        //std::array<key_type, 4> tempKeys = {default_value, default_value, default_value, default_value};
        __m512d key_d = _mm512_set_pd(
            static_cast<double>(*(it + 7)), static_cast<double>(*(it + 6)),
            static_cast<double>(*(it + 5)), static_cast<double>(*(it + 4)),
            static_cast<double>(*(it + 3)), static_cast<double>(*(it + 2)),
            static_cast<double>(*(it + 1)), static_cast<double>(*it)
        );
        result.pos.resize(keys_size); // pos등 error bound 제한을 준 결과값을 저장하는 공간 미리 할당//
        result.lo.resize(keys_size);
        result.hi.resize(keys_size);

        __m512i segment_ids_simd_ = base_type::get_segment_id_bySIMD(it);
        // layer 2 에서 위치를 예측할때 필요한 slope, intercept를 가져옴//
        __m512d slope = _mm512_set_pd(base_type::l2_[segment_ids_simd_[7]].slope_, base_type::l2_[segment_ids_simd_[6]].slope_
                                    , base_type::l2_[segment_ids_simd_[5]].slope_, base_type::l2_[segment_ids_simd_[4]].slope_
                                    , base_type::l2_[segment_ids_simd_[3]].slope_, base_type::l2_[segment_ids_simd_[2]].slope_
                                    , base_type::l2_[segment_ids_simd_[1]].slope_, base_type::l2_[segment_ids_simd_[0]].slope_);
        __m512d intercept = _mm512_set_pd(base_type::l2_[segment_ids_simd_[7]].intercept_, base_type::l2_[segment_ids_simd_[6]].intercept_
                                        , base_type::l2_[segment_ids_simd_[5]].intercept_, base_type::l2_[segment_ids_simd_[4]].intercept_
                                        , base_type::l2_[segment_ids_simd_[3]].intercept_, base_type::l2_[segment_ids_simd_[2]].intercept_
                                        , base_type::l2_[segment_ids_simd_[1]].intercept_, base_type::l2_[segment_ids_simd_[0]].intercept_);
        //fmadd함수로 key_d는 slope랑 곱하고 intercept랑 더 함//
        __m512d pred =  _mm512_roundscale_pd(_mm512_fmadd_pd(slope, key_d, intercept), _MM_FROUND_TO_ZERO);
        // pos(pred)를 uint64 simd변환 & 제한을 적용후 preds에 저장 //
        __m512i pred_i = _mm512_cvtpd_epi64(pred);
        __m512i min_values = _mm512_max_epi64(pred_i, base_type::zero);
        __m512i max_values = _mm512_min_epi64(base_type::n_keys_st, min_values);
        // error bound를 적용해서 pos가 0 인 특수상황을 처리하기 위한 과정 (즉, pos가 0이면, lo=0, 아니면 lo = (pred_i - error)//
        __m512i error_simd = _mm512_set_epi64(errors_[segment_ids_simd_[7]],errors_[segment_ids_simd_[6]]
                                             ,errors_[segment_ids_simd_[5]],errors_[segment_ids_simd_[4]]
                                             ,errors_[segment_ids_simd_[3]],errors_[segment_ids_simd_[2]]
                                             ,errors_[segment_ids_simd_[1]],errors_[segment_ids_simd_[0]]);
        __m512i lower_ = _mm512_sub_epi64(max_values, error_simd);
        __mmask8 compare_ = _mm512_cmpgt_epi64_mask(max_values, error_simd);
        __m512i lo_ = _mm512_mask_blend_epi64(compare_, base_type::zero, lower_);
        // error bound를 적용해서 pos+error가 n_keys_st보다 큰 특수상황을 처리하기 위한 과정 (즉, hi=n_keys_st, 아니면 hi = (pred_i + error)//
        __m512i upper_first = _mm512_add_epi64(max_values, error_simd);
        __m512i upper_second = _mm512_add_epi64(upper_first, _mm512_set1_epi64(1));
        __m512i hi_ = _mm512_min_epi64(upper_second, base_type::n_keys_st);

        _mm512_storeu_si512((__m512i*)result.pos.data(), max_values);
        _mm512_storeu_si512((__m512i*)result.lo.data(), lo_);
        _mm512_storeu_si512((__m512i*)result.hi.data(), hi_);
        return result;
    }

    /**
     * Returns the size of the index in bytes.
     * @return index size in bytes
     */
    std::size_t size_in_bytes() { return base_type::size_in_bytes() + errors_.size() * sizeof(errors_.front()); }

    std::size_t err_time() { return std::chrono::duration_cast<std::chrono::nanoseconds>(err_stop - err_start).count(); }
};


/**
 * Recursive model index with local individual bounds.
 */
template<typename Key, typename Layer1, typename Layer2>
class RmiLInd : public Rmi<Key, Layer1, Layer2>
{
    using base_type = Rmi<Key, Layer1, Layer2>;
    using key_type = Key;
    using layer1_type = Layer1;
    using layer2_type = Layer2;

    protected:
    /**
     * Struct to store a lower and an upper error bound.
     */
    std::chrono::time_point<std::chrono::steady_clock> err_start;
    std::chrono::time_point<std::chrono::steady_clock> err_stop; 

    struct bounds {
        std::size_t lo; ///< The lower error bound.
        std::size_t hi; ///< The upper error bound.

        /**
         * Default constructor.
         */
        bounds() : lo(0), hi(0) { }
    };

    std::vector<bounds> errors_; ///< The error bounds of the layer2 models.

    public:
    /**
     * Default constructor.
     */
    RmiLInd() = default;

    /**
     * Builds the index with @p layer2_size models in layer2 on the sorted @p keys.
     * @param keys vector of sorted keys to be indexed
     * @param layer2_size the number of models in layer2
     */
    RmiLInd(const std::vector<key_type> &keys, const std::size_t layer2_size, const std::size_t switch_n)
        : RmiLInd(keys.begin(), keys.end(), layer2_size, switch_n) { }

    /**
     * Builds the index with @p layer2_size models in layer2 on the sorted keys in the range [first, last).
     * @param first, last iterators that define the range of sorted keys to be indexed
     * @param layer2_size the number of models in layer2
     */
    template<typename RandomIt>
    RmiLInd(RandomIt first, RandomIt last, const std::size_t layer2_size, const std::size_t switch_n) : base_type(first, last, layer2_size, switch_n) {
        // Compute local individual errror bounds.
        err_start = std::chrono::steady_clock::now();
        errors_ = std::vector<bounds>(layer2_size);
        for (std::size_t i = 0; i != base_type::n_keys_; ++i) {
            key_type key = *(first + i);
            std::size_t segment_id = base_type::get_segment_id(key);
            std::size_t pred = std::clamp<double>(base_type::l2_[segment_id].predict(key), 0, base_type::n_keys_ - 1);
            if (pred > i) { // overestimation
                std::size_t &lo = errors_[segment_id].lo;
                lo = std::max(lo, pred - i);
            } else { // underestimation
                std::size_t &hi = errors_[segment_id].hi;
                hi = std::max(hi, i - pred);
            }
        }
        err_stop = std::chrono::steady_clock::now();
    }

    /**
     * Returns a position estimate and search bounds for a given key.
     * @param key to search for
     * @return position estimate and search bounds
     */
    Approx search(const key_type key) const {
        auto segment_id = base_type::get_segment_id(key);
        std::size_t pred = std::clamp<double>(base_type::l2_[segment_id].predict(key), 0, base_type::n_keys_ - 1);
        bounds err = errors_[segment_id];
        std::size_t lo = pred > err.lo ? pred - err.lo : 0;
        std::size_t hi = std::min(pred + err.hi + 1, base_type::n_keys_);
        return {pred, lo, hi};
    }

    Approx_SIMD search_bySIMD(std::vector<uint64_t>::const_iterator it) const {
        Approx_SIMD result;
        const std::size_t keys_size = 8;
        //std::array<key_type, 4> tempKeys = {default_value, default_value, default_value, default_value};
        __m512d key_d = _mm512_set_pd(
            static_cast<double>(*(it + 7)), static_cast<double>(*(it + 6)),
            static_cast<double>(*(it + 5)), static_cast<double>(*(it + 4)),
            static_cast<double>(*(it + 3)), static_cast<double>(*(it + 2)),
            static_cast<double>(*(it + 1)), static_cast<double>(*it)
        );
        result.pos.resize(keys_size); // pos등 error bound 제한을 준 결과값을 저장하는 공간 미리 할당//
        result.lo.resize(keys_size);
        result.hi.resize(keys_size);

        __m512i segment_ids_simd_ = base_type::get_segment_id_bySIMD(it);
        //__m256d segment_ids_double = _mm256_cvtepi64_pd(segment_ids_simd_);//
        // layer 2 에서 위치를 예측할때 필요한 slope, intercept를 가져옴//
        __m512d slope = _mm512_set_pd(base_type::l2_[segment_ids_simd_[7]].slope_, base_type::l2_[segment_ids_simd_[6]].slope_
                                    , base_type::l2_[segment_ids_simd_[5]].slope_, base_type::l2_[segment_ids_simd_[4]].slope_
                                    , base_type::l2_[segment_ids_simd_[3]].slope_, base_type::l2_[segment_ids_simd_[2]].slope_
                                    , base_type::l2_[segment_ids_simd_[1]].slope_, base_type::l2_[segment_ids_simd_[0]].slope_);
        __m512d intercept = _mm512_set_pd(base_type::l2_[segment_ids_simd_[7]].intercept_, base_type::l2_[segment_ids_simd_[6]].intercept_
                                        , base_type::l2_[segment_ids_simd_[5]].intercept_, base_type::l2_[segment_ids_simd_[4]].intercept_
                                        , base_type::l2_[segment_ids_simd_[3]].intercept_, base_type::l2_[segment_ids_simd_[2]].intercept_
                                        , base_type::l2_[segment_ids_simd_[1]].intercept_, base_type::l2_[segment_ids_simd_[0]].intercept_);
        //fmadd함수로 key_d는 slope랑 곱하고 intercept랑 더 함//
        __m512d pred =  _mm512_roundscale_pd(_mm512_fmadd_pd(slope, key_d, intercept), _MM_FROUND_TO_ZERO);
        // pos(pred)를 uint64 simd변환 & 제한을 적용후 preds에 저장 //
        __m512i pred_i = _mm512_cvtpd_epi64(pred);
        __m512i min_values = _mm512_max_epi64(pred_i, base_type::zero);
        __m512i max_values = _mm512_min_epi64(base_type::n_keys_st, min_values);
        // error bound를 적용해서 pos가 0 인 특수상황을 처리하기 위한 과정 (즉, pos가 0이면, lo=0, 아니면 lo = (pred_i - error_lo)//
        __m512i error_lo_simd = _mm512_set_epi64(errors_[segment_ids_simd_[7]].lo, errors_[segment_ids_simd_[6]].lo
                                                ,errors_[segment_ids_simd_[5]].lo, errors_[segment_ids_simd_[4]].lo
                                                ,errors_[segment_ids_simd_[3]].lo, errors_[segment_ids_simd_[2]].lo
                                                ,errors_[segment_ids_simd_[1]].lo, errors_[segment_ids_simd_[0]].lo);
        __m512i lower_ = _mm512_sub_epi64(max_values, error_lo_simd);
        __mmask8 compare_ = _mm512_cmpgt_epi64_mask(max_values, error_lo_simd);
        __m512i lo_ = _mm512_mask_blend_epi64(compare_, base_type::zero, lower_);
        // error bound를 적용해서 pos+error_hi가 n_keys_st보다 큰 특수상황을 처리하기 위한 과정 (즉, hi=n_keys_st, 아니면 hi = (pred_i + error_hi)//
        __m512i error_hi_simd = _mm512_set_epi64(errors_[segment_ids_simd_[7]].hi, errors_[segment_ids_simd_[6]].hi
                                                ,errors_[segment_ids_simd_[5]].hi, errors_[segment_ids_simd_[4]].hi
                                                ,errors_[segment_ids_simd_[3]].hi, errors_[segment_ids_simd_[2]].hi
                                                ,errors_[segment_ids_simd_[1]].hi, errors_[segment_ids_simd_[0]].hi);
        __m512i upper_first = _mm512_add_epi64(max_values, error_hi_simd);
        __m512i upper_second = _mm512_add_epi64(upper_first, _mm512_set1_epi64(1));
        __m512i hi_ = _mm512_min_epi64(upper_second, base_type::n_keys_st);

        _mm512_storeu_si512((__m512i*)result.pos.data(), max_values);
        _mm512_storeu_si512((__m512i*)result.lo.data(), lo_);
        _mm512_storeu_si512((__m512i*)result.hi.data(), hi_);
        return result;
    }
    

    /**
     * Returns the size of the index in bytes.
     * @return index size in bytes
     */
    std::size_t size_in_bytes() { return base_type::size_in_bytes() + errors_.size() * sizeof(errors_.front()); }

    std::size_t err_time() { return std::chrono::duration_cast<std::chrono::nanoseconds>(err_stop - err_start).count(); }
};
}// namespace rmi
