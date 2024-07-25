#pragma once

#include <cmath>
#include <x86intrin.h>
#include <immintrin.h>
#include "rmi/util/fn.hpp"
#include <chrono>

namespace rmi {

/**
 * A model that fits a linear segment from the first first to the last data point.
 *
 * We assume that x-values are sorted in ascending order and y-values are handed implicitly where @p offset and @p
 * offset + distance(first, last) are the first and last y-value, respectively. The y-values can be scaled by
 * providing a @p compression_factor.
 */
class LinearSpline
{
    private:

    public:
    double slope_;     ///< The slope of the linear segment.
    double intercept_; ///< The y-intercept of the lienar segment.
    __m512d slope_simd;
    __m512d intercept_simd;
    /**
     * Default contructor.
     */
    LinearSpline() = default;

    /**
     * Builds a linaer segment between the first and last data point.
     * @param first, last iterators to the first and last x-value the linear segment is fit on
     * @param offset first y-value the linear segment is fit on
     * @param compression_factor by which the y-values are scaled
     */
    template<typename RandomIt>
    LinearSpline(RandomIt first, RandomIt last, std::size_t offset = 0, double compression_factor = 1.f) {
        std::size_t n = std::distance(first, last);

        if (n == 0) {
            slope_ = 0.f;
            intercept_ = 0.f;
            return;
        }
        if (n == 1) {
            slope_ = 0.f;
            intercept_ = static_cast<double>(offset) * compression_factor;
            return;
        }

        double numerator = static_cast<double>(n); // (offset + n) - offset
        double denominator = static_cast<double>(*(last - 1) - *first);

        slope_ = denominator != 0.0 ? numerator/denominator * compression_factor : 0.0;
        intercept_ = offset * compression_factor - slope_ * *first;
        slope_simd = _mm512_set1_pd(slope_);
        intercept_simd =  _mm512_set1_pd(intercept_);

    }

    /**
     * Returns the estimated y-value of @p x.
     * @param x to estimate a y-value for
     * @return the estimated y-value for @p x
     */
    template<typename X>
    double predict(const X x) const { return std::fma(slope_, static_cast<double>(x), intercept_); }

    // predict_bySIMD
    __m512d predict_bySIMD(std::vector<uint64_t>::const_iterator it) const{
        __m512d key_d = _mm512_set_pd(
            static_cast<double>(*(it + 7)), static_cast<double>(*(it + 6)),
            static_cast<double>(*(it + 5)), static_cast<double>(*(it + 4)),
            static_cast<double>(*(it + 3)), static_cast<double>(*(it + 2)),
            static_cast<double>(*(it + 1)), static_cast<double>(*it)
        );
        __m512d result = _mm512_fmadd_pd(slope_simd, key_d, intercept_simd);

        return result;
    }

    __m512i predict_bySIMD_(__m512d & keys) const{
        return _mm512_cvtpd_epi64(_mm512_roundscale_pd(_mm512_fmadd_pd(slope_simd, keys, intercept_simd), _MM_FROUND_TO_ZERO));
    }

    /**
     * Returns the slope of the linear segment.
     * @return the slope of the linear segment
     */
    double slope() const { return slope_; }

    /**
     * Returns the y-intercept of the linear segment.
     * return the y-intercept of the linear segment
     */
    double intercept() const { return intercept_; }

    /**
     * Returns the size of the linear segment in bytes.
     * @return segment size in bytes.
     */
    std::size_t size_in_bytes() { return 2 * sizeof(double) + 2 * sizeof(__m512d); }

    /**
     * Writes the mathematical representation of the linear segment to an output stream.
     * @param out output stream to write the linear segment to
     * @param m the linear segment
     * @returns the output stream
     */
    friend std::ostream & operator<<(std::ostream &out, const LinearSpline &m) {
        return out << m.slope() << " * x + " << m.intercept();
    }
};


/**
 * A linear regression model that fits a straight line to minimize the mean squared error.
 *
 * We assume that x-values are sorted in ascending order and y-values are handed implicitly where @p offset and @p
 * offset + distance(first, last) are the first and last y-value, respectively. The y-values can be scaled by
 * providing a @p compression_factor.
 */
class LinearRegression
{
    private:

    // __m512d slope_simd_;  
    // __m512d intercept_simd_;
    // uint64_t seg_element_count_;

    public:
    double slope_;     ///< The slope of the linear function.
    double intercept_; ///< The y-intercept of the lienar function.
    /*
     * Default constructor.
     */
    LinearRegression() = default;

    /**
     * Builds a linaer regression model between on the given data points.
     * @param first, last iterators to the first and last x-value the linear regression is fit on
     * @param offset first y-value the linear regression is fit on
     * @param compression_factor by which the y-values are scaled
     */
    template<typename RandomIt>
    LinearRegression(RandomIt first, RandomIt last, std::size_t offset = 0, double compression_factor = 1.f) {
        // 시간복잡도 O(1)
        std::size_t n = std::distance(first, last);
        // 시간복잡도 O(1)
        if (n == 0) {
            slope_ = 0.f;
            intercept_ = 0.f;
            return;
        }
        if (n == 1) {
            slope_ = 0.f;
            intercept_ = static_cast<double>(offset) * compression_factor;
            return;
        }

        double mean_x = 0.0;
        double mean_y = 0.0;
        double c = 0.0;
        double m2 = 0.0;
        uint64_t i = 0;

        //시간 복잡도 O(8n)
        for (; i != n; ++i) {
            auto x = *(first + i);
            std::size_t y = offset + i;

            double dx = x - mean_x;
            mean_x += dx /  (i + 1);
            mean_y += (y - mean_y) / (i + 1);
            c += dx * (y - mean_y);

            double dx2 = x - mean_x;
            m2 += dx * dx2;
        }
        // 시간복잡도 O(1)
        double cov = c / (n - 1);
        double var = m2 / (n - 1);
        // 시간복잡도 O(1)
        if (var == 0.f) {
            slope_  = 0.f;
            intercept_ = mean_y;
            return;
        }
        slope_ = cov / var * compression_factor;
        intercept_ = mean_y * compression_factor - slope_ * mean_x;

        // 총 8n+4
    }

    /**
     * Returns the estimated y-value of @p x.
     * @param x to estimate a y-value for
     * @return the estimated y-value for @p x
     */
    template<typename X>
    double predict(const X x) const { return std::fma(slope_, static_cast<double>(x), intercept_); }

    // predict_bySIMD
    __m512d predict_bySIMD(std::vector<uint64_t>::const_iterator it) const{
        __m512d key_d = _mm512_set_pd(
            static_cast<double>(*(it + 7)), static_cast<double>(*(it + 6)),
            static_cast<double>(*(it + 5)), static_cast<double>(*(it + 4)),
            static_cast<double>(*(it + 3)), static_cast<double>(*(it + 2)),
            static_cast<double>(*(it + 1)), static_cast<double>(*it)
        );
        //__m512d result = _mm512_sub_pd(_mm512_fmadd_pd(slope_simd_, key_d, intercept_simd_), _mm512_set1_pd(0.5));
        __m512d result = _mm512_fmadd_pd(_mm512_set1_pd(slope_), key_d, _mm512_set1_pd(intercept_));
        return result;
    }

    /**
     * Returns the slope of the linear regression model.
     * @return the slope of the linear regression model
     */
    double slope() const { return slope_; }

    /**
     * Returns the y-intercept of the linear regression model.
     * return the y-intercept of the linear regression model
     */
    double intercept() const { return intercept_; }

    /**
     * Returns the size of the linear regression model in bytes.
     * @return model size in bytes.
     */
    std::size_t size_in_bytes() { return 2 * sizeof(double); }


    /**
     * Writes the mathematical representation of the linear regression model to an output stream.
     * @param out output stream to write the linear regression model to
     * @param m the linear regression model
     * @returns the output stream
     */
    friend std::ostream & operator<<(std::ostream &out, const LinearRegression &m) {
        return out << m.slope() << " * x + " << m.intercept();
    }
};


class LinearRegression_welford
{
    private:
    uint64_t training_time_;
    size_t remaining_elements;
    size_t key_n;

    public:
    double slope_;     ///< The slope of the linear function.
    double intercept_; ///< The y-intercept of the linear function.
    /*
     * Default constructor.
     */
    LinearRegression_welford() = default;
    
    /**
     * Builds a linear regression model between on the given data points.
     * @param first, last iterators to the first and last x-value the linear regression is fit on
     * @param offset first y-value the linear regression is fit on
     * @param compression_factor by which the y-values are scaled
     */
    template<typename RandomIt>
    LinearRegression_welford(RandomIt first, RandomIt last, std::size_t offset = 0, const std::size_t switch_n = 8, double compression_factor = 1.f) {
        std::size_t n = std::distance(first, last);
        // uint64_t key = *(last-1);
        // std::cout << std::fixed << std::setprecision(5) << key << " " << (double)key << std::endl;

        size_t remaining_elements = n % 8; //n이 8의 배수가 아닐경우
        uint64_t fragment  = n / 8; 
        
        __m512d mean_x = _mm512_setzero_pd();
        __m512d mean_y = _mm512_setzero_pd();
        __m512d x = _mm512_setzero_pd();
        __m512d y = _mm512_set_pd(8,7,6,5,4,3,2,1);        
        __m512d c = _mm512_setzero_pd();
        __m512d m2 = _mm512_setzero_pd();
        __m512d dx2 = _mm512_setzero_pd();
        __m512d dx = _mm512_setzero_pd();
        __m512d dy = _mm512_setzero_pd();
        __m512d reg_8 = _mm512_set1_pd(8); //레지스터에 +8을 위해
        __m512d reg_1 = _mm512_set1_pd(1); //레지스터에 +1을 위해
        __m512d reg_i = _mm512_set1_pd(1); //평균 계산할때 i로 나누기를 위해

        for (std::size_t i = 0; i < n-remaining_elements; i+=8) {
            x = _mm512_set_pd(static_cast<double>(*(first+i+7)),
                            static_cast<double>(*(first+i+6)),
                            static_cast<double>(*(first+i+5)),
                            static_cast<double>(*(first+i+4)),
                            static_cast<double>(*(first+i+3)),
                            static_cast<double>(*(first+i+2)),
                            static_cast<double>(*(first+i+1)),
                            static_cast<double>(*(first+i)));
            dx = _mm512_sub_pd(x, mean_x);
            mean_x = _mm512_add_pd(mean_x, _mm512_div_pd(dx, reg_i));

            dy = _mm512_sub_pd(y, mean_y);
            mean_y = _mm512_add_pd(mean_y, _mm512_div_pd(dy, reg_i));

            dy = _mm512_sub_pd(y, mean_y);
            c = _mm512_fmadd_pd(dx, dy, c);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

            dx2 = _mm512_sub_pd(x, mean_x);
            m2 = _mm512_fmadd_pd(dx, dx2, m2);      

            y = _mm512_add_pd(y, reg_8);
            reg_i = _mm512_add_pd(reg_i, reg_1);
        }

        double mean_x_array[8];
        double mean_y_array[8];
        double c_array[8];
        double m2_array[8];
        
        _mm512_storeu_pd(mean_x_array, mean_x);
        _mm512_storeu_pd(mean_y_array, mean_y);
        _mm512_storeu_pd(c_array, c);
        _mm512_storeu_pd(m2_array, m2);

        double j = 1.0;
        for(int i = 1; i<8; i++){
            j = i;
            mean_y_array[0] = (j * mean_y_array[0] + mean_y_array[i]) / (j+1);
            c_array[0]  = c_array[0] + c_array[i] + fragment * (mean_x_array[i] - mean_x_array[0]) * (mean_y_array[i] - (mean_y_array[0]));
            m2_array[0] = m2_array[0] + m2_array[i] + std::pow((mean_x_array[i] - mean_x_array[0]),2) * (j/(j+1)) * fragment;
            mean_x_array[0] = (j * mean_x_array[0] + mean_x_array[i]) / (j+1);
        }

        double X;
        double Y = (double)fragment*8;

        for (uint64_t i = n-remaining_elements; i < n; i++){
            X = *(first+i);
            double dx = X - mean_x_array[0];
            mean_x_array[0] += dx/(i+1);
            mean_y_array[0] += (Y-mean_y_array[0])/(i+1);
            c_array[0] += dx*(Y-mean_y_array[0]);

            double dx2 = X-mean_x_array[0];
            m2_array[0] += dx*dx2;
            Y++; 
        }

        mean_y_array[0] = mean_y_array[0] + offset - 1;
        
        if (m2_array[0] == 0.f) {
            slope_  = 0.f;
            intercept_ = mean_y_array[0];
            return;
        }

        slope_ = c_array[0] / m2_array[0] * compression_factor;
        intercept_ = mean_y_array[0] * compression_factor - slope_ * mean_x_array[0];

        // n+7
    }

    template<typename RandomIt>
    LinearRegression_welford(RandomIt first, RandomIt last, std::size_t offset = 0, int bit = 0, double compression_factor = 1.f){
        std::size_t n = std::distance(first, last);
        // std::cout << "Switch - SISD, n: "<< n << std::endl;
        if (n == 0) {
            slope_ = 0.f;
            intercept_ = 0.f;
            return;
        }
        if (n == 1) {
            slope_ = 0.f;
            intercept_ = static_cast<double>(offset) * compression_factor;
            return;
        }

        double mean_x = 0.0;
        double mean_y = 0.0;
        double c = 0.0;
        double m2 = 0.0;

        for (std::size_t i = 0; i != n; ++i) {
            auto x = *(first + i);
            std::size_t y = offset + i;

            double dx = x - mean_x;
            mean_x += dx /  (i + 1);
            mean_y += (y - mean_y) / (i + 1);
            c += dx * (y - mean_y);

            double dx2 = x - mean_x;
            m2 += dx * dx2;
        }

        double cov = c / (n - 1);
        double var = m2 / (n - 1);

        if (var == 0.f) {
            slope_  = 0.f;
            intercept_ = mean_y;
            return;
        }

        // std::cout << "mean_x: " << mean_x << std::endl;

        slope_ = cov / var * compression_factor;
        intercept_ = mean_y * compression_factor - slope_ * mean_x;

        // std::cout << "slope: " << slope_ << " intercept: " << intercept_ << std::endl;
    }

    /**
     * Returns the estimated y-value of @p x.
     * @param x to estimate a y-value for
     * @return the estimated y-value for @p x
     */
    template<typename X>
    double predict(const X x) const { return std::fma(slope_, static_cast<double>(x), intercept_); }

    __m512d predict_bySIMD(std::vector<uint64_t>::const_iterator it) const{
        __m512d key_d = _mm512_set_pd(
            static_cast<double>(*(it + 7)), static_cast<double>(*(it + 6)),
            static_cast<double>(*(it + 5)), static_cast<double>(*(it + 4)),
            static_cast<double>(*(it + 3)), static_cast<double>(*(it + 2)),
            static_cast<double>(*(it + 1)), static_cast<double>(*it)
        );
        //__m512d result = _mm512_sub_pd(_mm512_fmadd_pd(slope_simd_, key_d, intercept_simd_), _mm512_set1_pd(0.5));
        __m512d result = _mm512_fmadd_pd(_mm512_set1_pd(slope_), key_d, _mm512_set1_pd(intercept_));
        return result;
    }

    /**
     * Returns the slope of the linear regression model.
     * @return the slope of the linear regression model
     */
    double slope() const { return slope_; }

    /**
     * Returns the y-intercept of the linear regression model.
     * return the y-intercept of the linear regression model
     */
    double intercept() const { return intercept_; }

    /**
     * return the training time of the linear regression model
    */
    uint64_t training_time() const { return training_time_; }

    /**
     * Returns the size of the linear regression model in bytes.
     * @return model size in bytes.
     */
    std::size_t size_in_bytes() { return 2 * sizeof(double); }

    /**
     * Writes the mathematical representation of the linear regression model to an output stream.
     * @param out output stream to write the linear regression model to
     * @param m the linear regression model
     * @returns the output stream
     */
    friend std::ostream & operator<<(std::ostream &out, const LinearRegression_welford &m) {
        return out << m.slope() << " * x + " << m.intercept();
    }
};


class LinearRegression_float
{
    private:
    size_t key_n;

    public:
    double slope_;     ///< The slope of the linear function.
    double intercept_; ///< The y-intercept of the linear function.
    /*
     * Default constructor.
     */
    LinearRegression_float() = default;

    /**
     * Builds a linear regression model between on the given data points.
     * @param first, last iterators to the first and last x-value the linear regression is fit on
     * @param offset first y-value the linear regression is fit on
     * @param compression_factor by which the y-values are scaled
     */
    template<typename RandomIt>
    LinearRegression_float(RandomIt first, RandomIt last, std::size_t offset = 0, const std::size_t switch_n = 16, float compression_factor = 1.f) {
        std::size_t n = std::distance(first, last);
        key_n = n;
        // uint64_t key = *(last-1);
        // std::cout << std::fixed << std::setprecision(5) << key << " " << (double)key << std::endl;
        size_t remaining_elements = n % 16; //n이 16의 배수가 아닐경우
        uint64_t fragment  = n / 16; 
        
        __m512 mean_x = _mm512_setzero_ps();
        __m512 mean_y = _mm512_setzero_ps();
        __m512 x = _mm512_setzero_ps();
        __m512 y = _mm512_set_ps(16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);
        __m512 c = _mm512_setzero_ps();
        __m512 m2 = _mm512_setzero_ps();
        __m512 dx2 = _mm512_setzero_ps();
        __m512 dx = _mm512_setzero_ps();
        __m512 dy = _mm512_setzero_ps();
        __m512 reg_16 = _mm512_set1_ps(16); //레지스터에 +16을 위해
        __m512 reg_1 = _mm512_set1_ps(1); //레지스터에 +1을 위해
        __m512 reg_i = _mm512_set1_ps(1); //평균 계산할때 i로 나누기를 위해

        for (std::size_t i = 0; i < n-remaining_elements; i+=16) {
            x = _mm512_set_ps(static_cast<float>(*(first+i+15)), 
                            static_cast<float>(*(first+i+14)), 
                            static_cast<float>(*(first+i+13)), 
                            static_cast<float>(*(first+i+12)), 
                            static_cast<float>(*(first+i+11)), 
                            static_cast<float>(*(first+i+10)), 
                            static_cast<float>(*(first+i+9)), 
                            static_cast<float>(*(first+i+8)), 
                            static_cast<float>(*(first+i+7)), 
                            static_cast<float>(*(first+i+6)), 
                            static_cast<float>(*(first+i+5)), 
                            static_cast<float>(*(first+i+4)),
                            static_cast<float>(*(first+i+3)), 
                            static_cast<float>(*(first+i+2)),
                            static_cast<float>(*(first+i+1)), 
                            static_cast<float>(*(first+i))
                            );
            dx = _mm512_sub_ps(x, mean_x);
            mean_x = _mm512_add_ps(mean_x, _mm512_div_ps(dx, reg_i));

            dy = _mm512_sub_ps(y, mean_y);
            mean_y = _mm512_add_ps(mean_y, _mm512_div_ps(dy, reg_i));

            dy = _mm512_sub_ps(y, mean_y);
            c = _mm512_fmadd_ps(dx, dy, c);

            dx2 = _mm512_sub_ps(x, mean_x);
            m2 = _mm512_fmadd_ps(dx, dx2, m2);

            y = _mm512_add_ps(y, reg_16);
            reg_i = _mm512_add_ps(reg_i, reg_1);
        }

        float mean_x_array[16];
        float mean_y_array[16];
        float c_array[16];
        float m2_array[16];

        // print_reg(m2);

        _mm512_storeu_ps(mean_x_array, mean_x);
        _mm512_storeu_ps(mean_y_array, mean_y);
        _mm512_storeu_ps(c_array, c);
        _mm512_storeu_ps(m2_array, m2);

        float j = 1.0;
        for(int i = 1; i<16; i++){
            j = i;
            mean_y_array[0] = (j * mean_y_array[0] + mean_y_array[i]) / (j+1);
            c_array[0]  = c_array[0] + c_array[i] + fragment * (mean_x_array[i] - mean_x_array[0]) * (mean_y_array[i] - (mean_y_array[0]));
            m2_array[0] = m2_array[0] + m2_array[i] + std::pow((mean_x_array[i] - mean_x_array[0]),2) * (j/(j+1)) * fragment;
            mean_x_array[0] = (j * mean_x_array[0] + mean_x_array[i]) / (j+1);
        }

        float X;
        float Y = (float)fragment*16;

        for (uint64_t i = n-remaining_elements; i < n; i++){
            X = *(first+i);
            float dx = X - mean_x_array[0];
            mean_x_array[0] += dx/(i+1);
            mean_y_array[0] += (Y-mean_y_array[0])/(i+1);
            c_array[0] += dx*(Y-mean_y_array[0]);

            float dx2 = X-mean_x_array[0];
            m2_array[0] += dx*dx2;
            Y++; 
        }

        mean_y_array[0] = mean_y_array[0] + offset - 1;

        if (m2_array[0] == 0.f) {
            slope_  = 0.f;
            intercept_ = mean_y_array[0];
            return;
        }

        slope_ = c_array[0] / m2_array[0] * compression_factor;
        intercept_ = mean_y_array[0] * compression_factor - slope_ * mean_x_array[0];

        // std::cout << "mean_x: " << mean_x_array[0] << " mean_y: " << mean_y_array[0] << std::endl;
        // std::cout << "cov: " << c_array[0] << " var: " << m2_array[0] << std::endl;

        // std::cout << "slope: " << slope_ << " intercept: " << intercept_ << std::endl;
        // std::cout << std::endl;
    }
    template<typename RandomIt>
    LinearRegression_float(RandomIt first, RandomIt last, std::size_t offset = 0, int bit = 0, double compression_factor = 1.f){
        std::size_t n = std::distance(first, last);
        // std::cout << "Switch - SISD, n: "<< n << std::endl;
        if (n == 0) {
            slope_ = 0.f;
            intercept_ = 0.f;
            return;
        }
        if (n == 1) {
            slope_ = 0.f;
            intercept_ = static_cast<double>(offset) * compression_factor;
            return;
        }

        double mean_x = 0.0;
        double mean_y = 0.0;
        double c = 0.0;
        double m2 = 0.0;

        for (std::size_t i = 0; i != n; ++i) {
            auto x = *(first + i);
            std::size_t y = offset + i;

            double dx = x - mean_x;
            mean_x += dx /  (i + 1);
            mean_y += (y - mean_y) / (i + 1);
            c += dx * (y - mean_y);

            double dx2 = x - mean_x;
            m2 += dx * dx2;
        }

        double cov = c / (n - 1);
        double var = m2 / (n - 1);

        if (var == 0.f) {
            slope_  = 0.f;
            intercept_ = mean_y;
            return;
        }

        // std::cout << "mean_x: " << mean_x << std::endl;

        slope_ = cov / var * compression_factor;
        intercept_ = mean_y * compression_factor - slope_ * mean_x;

        // std::cout << "slope: " << slope_ << " intercept: " << intercept_ << std::endl;
    }

    void print_reg(__m512 reg){
        float key_values[8];
        _mm512_storeu_ps(key_values, reg);
        std::cout << "AVX Register: ";
        for (int j = 0; j < 8; ++j) {
            std::cout << key_values[j] << " ";
        }
        std::cout << std::endl;
    }
    void print_reg_i(__m512i reg){
        double key_values[8];
        _mm512_storeu_si512(key_values, reg);
        std::cout << "AVX Register: ";
        for (int j = 0; j < 8; ++j) {
            std::cout << key_values[j] << " ";
        }
        std::cout << std::endl;
    }
    

    /**
     * Returns the estimated y-value of @p x.
     * @param x to estimate a y-value for
     * @return the estimated y-value for @p x
     */
    template<typename X>
    double predict(const X x) const { return std::fma(slope_, static_cast<double>(x), intercept_); }

    __m512d predict_bySIMD(std::vector<uint64_t>::const_iterator it) const{
        __m512d key_d = _mm512_set_pd(
            static_cast<double>(*(it + 7)), static_cast<double>(*(it + 6)),
            static_cast<double>(*(it + 5)), static_cast<double>(*(it + 4)),
            static_cast<double>(*(it + 3)), static_cast<double>(*(it + 2)),
            static_cast<double>(*(it + 1)), static_cast<double>(*it)
        );
        //__m512d result = _mm512_sub_pd(_mm512_fmadd_pd(slope_simd_, key_d, intercept_simd_), _mm512_set1_pd(0.5));
        __m512d result = _mm512_fmadd_pd(_mm512_set1_pd(slope_), key_d, _mm512_set1_pd(intercept_));
        return result;
    }

    /**
     * Returns the slope of the linear regression model.
     * @return the slope of the linear regression model
     */
    double slope() const { return slope_; }

    /**
     * Returns the y-intercept of the linear regression model.
     * return the y-intercept of the linear regression model
     */
    double intercept() const { return intercept_; }

    /**
     * Returns the size of the linear regression model in bytes.
     * @return model size in bytes.
     */
    std::size_t size_in_bytes() { return 2 * sizeof(double); }

    /**
     * Writes the mathematical representation of the linear regression model to an output stream.
     * @param out output stream to write the linear regression model to
     * @param m the linear regression model
     * @returns the output stream
     */
    friend std::ostream & operator<<(std::ostream &out, const LinearRegression_float &m) {
        return out << m.slope() << " * x + " << m.intercept();
    }
};


/**
 * A model that fits a cubic segment from the first first to the last data point.
 *
 * We assume that x-values are sorted in ascending order and y-values are handed implicitly where @p offset and @p
 * offset + distance(first, last) are the first and last y-value, respectively. The y-values can be scaled by
 * providing a @p compression_factor.
 */
class CubicSpline
{
    private:
    double a_; ///< The cubic coefficient.
    double b_; ///< The quadric coefficietn.
    double c_; ///< The linear coefficient.
    double d_; ///< The y-intercept.

    public:
    /**
     * Default constructor.
     */
    CubicSpline() = default;

    /**
     * Builds a cubic segment between the first and last data point.
     * @param first, last iterators to the first and last x-value the cubic segment is fit on
     * @param offset first y-value the cubic segment is fit on
     * @param compression_factor by which the y-values are scaled
     */
    template<typename RandomIt>
    CubicSpline(RandomIt first, RandomIt last, std::size_t offset = 0, double compression_factor = 1.f) {
        std::size_t n = std::distance(first, last);

        if (n == 0) {
            a_ = 0.f;
            b_ = 0.f;
            c_ = 1.f;
            d_ = 0.f;
            return;
        }
        if (n == 1 or *first == *(last - 1)) {
            a_ = 0.f;
            b_ = 0.f;
            c_ = 0.f;
            d_ = static_cast<double>(offset) * compression_factor;
            return;
        }

        double xmin = static_cast<double>(*first);
        double ymin = static_cast<double>(offset) * compression_factor;
        double xmax = static_cast<double>(*(last - 1));
        double ymax = static_cast<double>(offset + n - 1) * compression_factor;

        double x1 = 0.0;
        double y1 = 0.0;
        double x2 = 1.0;
        double y2 = 1.0;

        double sxn, syn = 0.0;
        for (std::size_t i = 0; i != n; ++i) {
            double x = static_cast<double>(*(first + i));
            double y = static_cast<double>(offset + i) * compression_factor;
            sxn = (x - xmin) / (xmax - xmin);
            if (sxn > 0.0) {
                syn = (y - ymin) / (ymax - ymin);
                break;
            }
        }
        double m1 = (syn - y1) / (sxn - x1);

        double sxp, syp = 0.0;
        for (std::size_t i = 0; i != n; ++i) {
            double x = static_cast<double>(*(first + i));
            double y = static_cast<double>(offset + i) * compression_factor;
            sxp = (x - xmin) / (xmax - xmin);
            if (sxp < 1.0) {
                syp = (y - ymin) / (ymax - ymin);
                break;
            }
        }
        double m2 = (y2 - syp) / (x2 - sxp);

        if (std::pow(m1, 2.0) + std::pow(m2, 2.0) > 9.0) {
            double tau = 3.0 / std::sqrt(std::pow(m1, 2.0) + std::pow(m2, 2.0));
            m1 *= tau;
            m2 *= tau;
        }

        a_ = (m1 + m2 - 2.0)
            / std::pow(xmax - xmin, 3.0);

        b_ = -(xmax * (2.0 * m1 + m2 - 3.0) + xmin * (m1 + 2.0 * m2 - 3.0))
            / std::pow(xmax - xmin, 3.0);

        c_ = (m1 * std::pow(xmax, 2.0) + m2 * std::pow(xmin, 2.0) + xmax * xmin * (2.0 * m1 + 2.0 * m2 - 6.0))
            / std::pow(xmax - xmin, 3.0);

        d_ = -xmin * (m1 * std::pow(xmax, 2.0) + xmax * xmin * (m2 - 3.0) + std::pow(xmin, 2.0))
            / std::pow(xmax - xmin, 3.0);

        a_ *= ymax - ymin;
        b_ *= ymax - ymin;
        c_ *= ymax - ymin;
        d_ *= ymax - ymin;
        d_ += ymin;

        // Check if linear spline performs better.
        // LinearSpline ls(first, last, offset, compression_factor);

        // double ls_error = 0.f;
        // double cs_error = 0.f;

        // for (std::size_t i = 0; i != n; ++i) {
        //     double y = (offset +i) * compression_factor;
        //     auto key = *(first + i);
        //     double ls_pred = ls.predict(key);
        //     double cs_pred = predict(key);
        //     ls_error += std::abs(ls_pred - y);
        //     cs_error += std::abs(cs_pred - y);
        // }

        // if (ls_error < cs_error) {
        //     a_ = 0;
        //     b_ = 0;
        //     c_ = ls.slope();
        //     d_ = ls.intercept();
        // }
    }

    /**
     * Returns the estimated y-value of @p x.
     * @param x to estimate a y-value for
     * @return the estimated y-value for @p x
     */
    template<typename X>
    double predict(const X x) const {
        double x_ = static_cast<double>(x);
        double v1 = std::fma(a_, x_, b_);
        double v2 = std::fma(v1, x_, c_);
        double v3 = std::fma(v2, x_, d_);
        return v3;
    }

    __m512d predict_bySIMD(const uint64_t* keys) const{
        __m512d a = _mm512_set_pd(a_, a_, a_, a_, a_, a_, a_, a_);
        __m512d b = _mm512_set_pd(b_, b_, b_, b_, b_, b_, b_, b_);
        __m512d c = _mm512_set_pd(c_, c_, c_, c_, c_, c_, c_, c_);
        __m512d d = _mm512_set_pd(d_, d_, d_, d_, d_, d_, d_, d_);
        __m512d key_d = _mm512_set_pd(
            static_cast<double>(keys[7]), static_cast<double>(keys[6]),
            static_cast<double>(keys[5]), static_cast<double>(keys[4]),
            static_cast<double>(keys[3]), static_cast<double>(keys[2]),
            static_cast<double>(keys[1]), static_cast<double>(keys[0])
        );
        __m512d x_ = _mm512_set_pd(key_d[7], key_d[6], key_d[5], key_d[4], key_d[3], key_d[2], key_d[1], key_d[0]);
        __m512d v1 = _mm512_fmadd_pd(a, x_, b);
        __m512d v2 = _mm512_fmadd_pd(v1, x_, c);
        __m512d v3 = _mm512_fmadd_pd(v2, x_, d);
        return v3;
    }

    /** Returns the cubic coefficient.
     * @return the cubic coefficient
     */
    double a() const { return a_; }

    /** Returns the quadric coefficient.
     * @return the quadric coefficient
     */
    double b() const { return b_; }

    /** Returns the linear coefficient.
     * @return the linear coefficient
     */
    double c() const { return c_; }

    /** Returns the y-intercept.
     * @return the y-intercept
     */
    double d() const { return d_; }

    /**
     * Returns the size of the cubic segment in bytes.
     * @return segment size in bytes.
     */
    std::size_t size_in_bytes() { return 4 * sizeof(double); }

    /**
     * Writes the mathematical representation of the cubic segment to an output stream.
     * @param out output stream to write the cubic segment to
     * @param m the cubic segment
     * @returns the output stream
     */
    friend std::ostream & operator<<(std::ostream &out, const CubicSpline &m) {
        return out << m.a() << " * x^3 + "
                   << m.b() << " * x^2 + "
                   << m.c() << " * x + d";
    }
};


/**
 * A radix model that projects a x-values to their most significant bits after eliminating the common prefix.
 *
 * We assume that x-values are sorted in ascending order and y-values are handed implicitly where @p offset and @p
 * offset + distance(first, last) are the first and last y-value, respectively. The y-values can be scaled by
 * providing a @p compression_factor.
 *
 * @tparam the type of x-values.
 */
template<typename X = uint64_t>
class Radix
{
    using x_type = X;

    private:
    x_type mask_; ///< The mask for parallel bits extract.

    public:
    /*
     * Default constructor.
     */
    Radix() = default;

    /**
     * Builds a radix model on the given data points.
     * @param first, last iterators to the first and last x-value the linear regression is fit on
     * @param offset first y-value the linear regression is fit on
     * @param compression_factor by which the y-values are scaled
     */
    template<typename RandomIt>
    Radix(RandomIt first, RandomIt last, std::size_t offset = 0, double compression_factor = 1.f) {
        std::size_t n = std::distance(first, last);

        if (n == 0) {
            mask_ = 0;
            return;
        }

        auto prefix = common_prefix_width(*first, *(last - 1)); // compute common prefix length

        if (prefix == (sizeof(x_type) * 8)) {
            mask_ = 42; // TODO: What should the mask be in this case?
            return;
        }

        // Determine radix width.
        std::size_t max = static_cast<std::size_t>(offset + n - 1) * compression_factor;
        bool is_mersenne = (max & (max + 1)) == 0; // check if max is 2^n-1
        auto radix = is_mersenne ? bit_width<std::size_t>(max) : bit_width<std::size_t>(max) - 1;

        // Mask all bits but the radix
        mask_ = (~(x_type)0 >> prefix) & (~(x_type)0 << ((sizeof(x_type) * 8) - radix - prefix)); //0xffff << prefix_
    }

    /**
     * Returns the estimated y-value of @p x.
     * @param x to estimate a y-value for
     * @return the estimated y-value for @p x
     */
    // double predict(const x_type x) const { return (x << prefix_) >> ((sizeof(x_type) * 8) - radix_); }
    double predict(const x_type x) const {
        if constexpr(sizeof(x_type) <= sizeof(unsigned)) {
            return _pext_u32(x, mask_);
        } else if constexpr(sizeof(x_type) <= sizeof(unsigned long long)) {
            return _pext_u64(x, mask_);
        } else {
            static_assert(sizeof(x_type) > sizeof(unsigned long long), "unsupported width of integral type");
        }
    }


    __m512d predict_bySIMD(const uint64_t* keys) const {
        double results[8];
        for (std::size_t i = 0; i < 8; i++) {
            if constexpr(sizeof(x_type) <= sizeof(unsigned)) {
                results[i] = static_cast<double>(_pext_u32(static_cast<unsigned>(keys[i]), static_cast<unsigned>(mask_)));
            } else if constexpr(sizeof(x_type) <= sizeof(unsigned long long)) {
                results[i] = static_cast<double>(_pext_u64(keys[i], mask_));
            }
        }

        __m512d mask = _mm512_set_pd(results[7], results[6], results[5], results[4], results[3], results[2], results[1], results[0]);
        return mask;
    }


    /**
     * Returns the mask used for parallel bits extraction.
     * @return the mask
     */
    uint8_t mask() const { return mask_; }

    /**
     * Returns the size of the radix model in bytes.
     * @return radix model size in bytes.
     */
    std::size_t size_in_bytes() { return sizeof(mask_); }

    /**
     * Writes a human readable representation of the radix model to an output stream.
     * @param out output stream to write the radix model to
     * @param m the radix model
     * @returns the output stream
     */
    friend std::ostream & operator<<(std::ostream &out, const Radix &m) {
        return out << "_pext(x, " << m.mask() << ")";
    }
};

} // namespace rmi