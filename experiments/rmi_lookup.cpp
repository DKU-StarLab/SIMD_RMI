#include <chrono>
#include <random>

#include "argparse/argparse.hpp"

#include "rmi/models.hpp"
#include "rmi/rmi.hpp"
#include "rmi/util/fn.hpp"
#include "rmi/util/search.hpp"
#include <iomanip>

#include <iostream>
#include <cstdint>

using key_type = uint64_t;
using namespace std::chrono;

std::size_t s_glob; ///< global size_t variable
constexpr std::size_t alignment = 64; // AVX-512 的对齐要求
constexpr std::size_t group_size = 8; // 每组键的大小

/**
 * Measures lookup times of @p samples on a given @p Rmi and writes results to `std::cout`.
 * @tparam Key key type
 * @tparam Rmi RMI type
 * @tparam Search search type
 * @param keys on which the RMI is built
 * @param n_models number of models in the second layer of the RMI
 * @param samples for which the lookup time is measured
 * @param n_reps number of repetitions
 * @param dataset_name name of the dataset
 * @param layer1 model type of the first layer
 * @param layer2 model type of the second layer
 * @param bound_type used by the RMI
 * @param search used by the RMI for correction prediction errors
 */

inline uint64_t rdtsc() {
    uint32_t lo, hi;
    __asm__ __volatile__ (
        "rdtsc" : "=a" (lo), "=d" (hi)
    );
    return ((uint64_t)hi << 32) | lo;
}

template<typename Key, typename Rmi, typename Search>
void experiment(const std::vector<key_type> &keys,
                const std::size_t n_models,
                const std::vector<key_type> &samples,
                const std::size_t n_reps,
                const std::string dataset_name,
                const std::string layer1,
                const std::string layer2,
                const std::string bound_type,
                const std::string search,
                const std::size_t switch_n)
{
    using rmi_type = Rmi;
    auto search_fn = Search();

    //key_type* key_group = static_cast<key_type*>(std::aligned_alloc(alignment, group_size * sizeof(key_type)));

    // Build RMI.
    rmi_type rmi(keys, n_models, switch_n);

    // Perform n_reps runs.
    //auto start = steady_clock::now();
    for (std::size_t rep = 0; rep != n_reps; ++rep) {

        // Lookup time.
        double lookup_accu = 0;
        std::size_t predict_time = 0;
        std::size_t correct_time = 0;
        std::size_t lookup_time = 0;
        auto it = samples.begin();
        auto start = steady_clock::now();
        for (std::size_t i = 0; i < samples.size(); i += group_size) {
            //auto start = steady_clock::now();
            //auto it = samples.begin()+i;
            auto predict_start = steady_clock::now();
            //auto predict_start = rdtsc();
            auto range = rmi.search_bySIMD(it+i);
            //auto predict_stop = rdtsc();
            auto predict_stop = steady_clock::now();
            // if(i == 0)
            // {
            //     for(int i = 0; i < n_models; ++i)
            //     {
            //         std::cout << range.err[i] << std::endl;
            //     }
            // }

            auto correct_start = steady_clock::now();
            auto pos = search_fn(keys.begin() + range.lo[0], keys.begin() + range.hi[0], keys.begin() + range.pos[0], *(it+i));
            lookup_accu += *pos == *(it+i) ? 1 : 0;//

            pos = search_fn(keys.begin() + range.lo[1], keys.begin() + range.hi[1], keys.begin() + range.pos[1], *(it+i+1));
            lookup_accu += *pos == *(it+i+1) ? 1 : 0;//

            pos = search_fn(keys.begin() + range.lo[2], keys.begin() + range.hi[2], keys.begin() + range.pos[2], *(it+i+2));
            lookup_accu += *pos == *(it+i+2) ? 1 : 0;//

            pos = search_fn(keys.begin() + range.lo[3], keys.begin() + range.hi[3], keys.begin() + range.pos[3], *(it+i+3));
            lookup_accu += *pos == *(it+i+3) ? 1 : 0;//

            pos = search_fn(keys.begin() + range.lo[4], keys.begin() + range.hi[4], keys.begin() + range.pos[4], *(it+i+4));
            lookup_accu += *pos == *(it+i+4) ? 1 : 0;//

            pos = search_fn(keys.begin() + range.lo[5], keys.begin() + range.hi[5], keys.begin() + range.pos[5], *(it+i+5));
            lookup_accu += *pos == *(it+i+5) ? 1 : 0;//

            pos = search_fn(keys.begin() + range.lo[6], keys.begin() + range.hi[6], keys.begin() + range.pos[6], *(it+i+6));
            lookup_accu += *pos == *(it+i+6) ? 1 : 0;//

            pos = search_fn(keys.begin() + range.lo[7], keys.begin() + range.hi[7], keys.begin() + range.pos[7], *(it+i+7));
            lookup_accu += *pos == *(it+i+7) ? 1 : 0;//
            auto correct_stop = steady_clock::now();

            //std::cout << range.err[i] << std::endl;

            //auto stop = steady_clock::now();
            predict_time += std::chrono::duration_cast<std::chrono::nanoseconds>(predict_stop - predict_start).count();
            //predict_time += (predict_stop - predict_start);
            correct_time += std::chrono::duration_cast<std::chrono::nanoseconds>(correct_stop - correct_start).count();
        }

        auto stop = steady_clock::now();
        lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

        std::size_t lookup_latency = lookup_time / samples.size();
        std::size_t predict_latency = predict_time / samples.size();
        std::size_t correct_latency = correct_time / samples.size();
        lookup_accu = lookup_accu / samples.size();
        s_glob = lookup_accu;

        // Report results.
                  // Dataset
        std::cout << dataset_name << ','
                  << keys.size() << ','
                  // Index
                  << layer1 << ','
                  << layer2 << ','
                  << n_models << ','
                  << switch_n << ','
                  << bound_type << ','
                  << search << ','
                  << rmi.size_in_bytes() << ','
                  // Experiment
                  << rep << ','
                  << samples.size() << ','
                  // Results
                  << lookup_latency << ','
                  << predict_latency << ','
                  << correct_latency << ','
                  // Checksums
                  << std::fixed << std::setprecision(7) << lookup_accu << ','
                  << std::endl;
    } // reps

}


/**
 * @brief experiment function pointer
 */
typedef void (*exp_fn_ptr)(const std::vector<key_type>&,
                           const std::size_t,
                           const std::vector<key_type>&,
                           const std::size_t,
                           const std::string,
                           const std::string,
                           const std::string,
                           const std::string,
                           const std::string,
                           const std::size_t);

/**
 * RMI configuration that holds the string representation of model types of layer 1 and layer 2, error bound type, and
 * search algorithm.
 */
struct Config {
    std::string layer1;
    std::string layer2;
    std::string bound_type;
    std::string search;
};

/**
 * Comparator class for @p Config objects.
 */
struct ConfigCompare {
    bool operator() (const Config &lhs, const Config &rhs) const {
        if (lhs.layer1 != rhs.layer1) return lhs.layer1 < rhs.layer1;
        if (lhs.layer2 != rhs.layer2) return lhs.layer2 < rhs.layer2;
        if (lhs.bound_type != rhs.bound_type) return lhs.bound_type < rhs.bound_type;
        return lhs.search < rhs.search;
    }
};

#define ENTRIES(L1, L2, LT1, LT2) \
    { {#L1, #L2, "none", "binary"}, &experiment<key_type, rmi::Rmi<key_type, LT1, LT2>, BinarySearch> }, \
    { {#L1, #L2, "labs", "binary"}, &experiment<key_type, rmi::RmiLAbs<key_type, LT1, LT2>, BinarySearch> }, \
    { {#L1, #L2, "lind", "binary"}, &experiment<key_type, rmi::RmiLInd<key_type, LT1, LT2>, BinarySearch> }, \
    { {#L1, #L2, "gabs", "binary"}, &experiment<key_type, rmi::RmiGAbs<key_type, LT1, LT2>, BinarySearch> }, \
    { {#L1, #L2, "gind", "binary"}, &experiment<key_type, rmi::RmiGInd<key_type, LT1, LT2>, BinarySearch> }, \
    { {#L1, #L2, "none", "binary_branchless"}, &experiment<key_type, rmi::Rmi<key_type, LT1, LT2>, BinarySearch_Branchless> }, \
    { {#L1, #L2, "lind", "binary_branchless"}, &experiment<key_type, rmi::RmiLInd<key_type, LT1, LT2>, BinarySearch_Branchless> }, \
    { {#L1, #L2, "gabs", "binary_branchless"}, &experiment<key_type, rmi::RmiGAbs<key_type, LT1, LT2>, BinarySearch_Branchless> }, \
    { {#L1, #L2, "gind", "binary_branchless"}, &experiment<key_type, rmi::RmiGInd<key_type, LT1, LT2>, BinarySearch_Branchless> }, \
    { {#L1, #L2, "labs", "binary_branchless"}, &experiment<key_type, rmi::RmiLAbs<key_type, LT1, LT2>, BinarySearch_Branchless> }, \
    { {#L1, #L2, "none", "model_biased_binary"}, &experiment<key_type, rmi::Rmi<key_type, LT1, LT2>, ModelBiasedBinarySearch> }, \
    { {#L1, #L2, "labs", "model_biased_binary"}, &experiment<key_type, rmi::RmiLAbs<key_type, LT1, LT2>, ModelBiasedBinarySearch> }, \
    { {#L1, #L2, "lind", "model_biased_binary"}, &experiment<key_type, rmi::RmiLInd<key_type, LT1, LT2>, ModelBiasedBinarySearch> }, \
    { {#L1, #L2, "gabs", "model_biased_binary"}, &experiment<key_type, rmi::RmiGAbs<key_type, LT1, LT2>, ModelBiasedBinarySearch> }, \
    { {#L1, #L2, "gind", "model_biased_binary"}, &experiment<key_type, rmi::RmiGInd<key_type, LT1, LT2>, ModelBiasedBinarySearch> }, \
    { {#L1, #L2, "none", "linear"}, &experiment<key_type, rmi::Rmi<key_type, LT1, LT2>, LinearSearch> }, \
    { {#L1, #L2, "labs", "linear"}, &experiment<key_type, rmi::RmiLAbs<key_type, LT1, LT2>, LinearSearch> }, \
    { {#L1, #L2, "lind", "linear"}, &experiment<key_type, rmi::RmiLInd<key_type, LT1, LT2>, LinearSearch> }, \
    { {#L1, #L2, "gabs", "linear"}, &experiment<key_type, rmi::RmiGAbs<key_type, LT1, LT2>, LinearSearch> }, \
    { {#L1, #L2, "gind", "linear"}, &experiment<key_type, rmi::RmiGInd<key_type, LT1, LT2>, LinearSearch> }, \
    { {#L1, #L2, "none", "model_biased_linear"}, &experiment<key_type, rmi::Rmi<key_type, LT1, LT2>, ModelBiasedLinearSearch> }, \
    { {#L1, #L2, "labs", "model_biased_linear"}, &experiment<key_type, rmi::RmiLAbs<key_type, LT1, LT2>, ModelBiasedLinearSearch> }, \
    { {#L1, #L2, "lind", "model_biased_linear"}, &experiment<key_type, rmi::RmiLInd<key_type, LT1, LT2>, ModelBiasedLinearSearch> }, \
    { {#L1, #L2, "gabs", "model_biased_linear"}, &experiment<key_type, rmi::RmiGAbs<key_type, LT1, LT2>, ModelBiasedLinearSearch> }, \
    { {#L1, #L2, "gind", "model_biased_linear"}, &experiment<key_type, rmi::RmiGInd<key_type, LT1, LT2>, ModelBiasedLinearSearch> }, \
    { {#L1, #L2, "none", "exponential"}, &experiment<key_type, rmi::Rmi<key_type, LT1, LT2>, ExponentialSearch> }, \
    { {#L1, #L2, "labs", "exponential"}, &experiment<key_type, rmi::RmiLAbs<key_type, LT1, LT2>, ExponentialSearch> }, \
    { {#L1, #L2, "lind", "exponential"}, &experiment<key_type, rmi::RmiLInd<key_type, LT1, LT2>, ExponentialSearch> }, \
    { {#L1, #L2, "gabs", "exponential"}, &experiment<key_type, rmi::RmiGAbs<key_type, LT1, LT2>, ExponentialSearch> }, \
    { {#L1, #L2, "gind", "exponential"}, &experiment<key_type, rmi::RmiGInd<key_type, LT1, LT2>, ExponentialSearch> }, \
    { {#L1, #L2, "none", "model_biased_exponential"}, &experiment<key_type, rmi::Rmi<key_type, LT1, LT2>, ModelBiasedExponentialSearch> }, \
    { {#L1, #L2, "labs", "model_biased_exponential"}, &experiment<key_type, rmi::RmiLAbs<key_type, LT1, LT2>, ModelBiasedExponentialSearch> }, \
    { {#L1, #L2, "lind", "model_biased_exponential"}, &experiment<key_type, rmi::RmiLInd<key_type, LT1, LT2>, ModelBiasedExponentialSearch> }, \
    { {#L1, #L2, "gabs", "model_biased_exponential"}, &experiment<key_type, rmi::RmiGAbs<key_type, LT1, LT2>, ModelBiasedExponentialSearch> }, \
    { {#L1, #L2, "gind", "model_biased_exponential"}, &experiment<key_type, rmi::RmiGInd<key_type, LT1, LT2>, ModelBiasedExponentialSearch> }, \
    { {#L1, #L2, "none", "linear_simd"}, &experiment<key_type, rmi::Rmi<key_type, LT1, LT2>, LinearSearch_SIMD> }, \
    { {#L1, #L2, "labs", "linear_simd"}, &experiment<key_type, rmi::RmiLAbs<key_type, LT1, LT2>, LinearSearch_SIMD> }, \
    { {#L1, #L2, "lind", "linear_simd"}, &experiment<key_type, rmi::RmiLInd<key_type, LT1, LT2>, LinearSearch_SIMD> }, \
    { {#L1, #L2, "gabs", "linear_simd"}, &experiment<key_type, rmi::RmiGAbs<key_type, LT1, LT2>, LinearSearch_SIMD> }, \
    { {#L1, #L2, "gind", "linear_simd"}, &experiment<key_type, rmi::RmiGInd<key_type, LT1, LT2>, LinearSearch_SIMD> }, \
    { {#L1, #L2, "none", "model_biased_linear_simd"}, &experiment<key_type, rmi::Rmi<key_type, LT1, LT2>, ModelBiasedLinearSearch_SIMD> }, \
    { {#L1, #L2, "labs", "model_biased_linear_simd"}, &experiment<key_type, rmi::RmiLAbs<key_type, LT1, LT2>, ModelBiasedLinearSearch_SIMD> }, \
    { {#L1, #L2, "lind", "model_biased_linear_simd"}, &experiment<key_type, rmi::RmiLInd<key_type, LT1, LT2>, ModelBiasedLinearSearch_SIMD> }, \
    { {#L1, #L2, "gabs", "model_biased_linear_simd"}, &experiment<key_type, rmi::RmiGAbs<key_type, LT1, LT2>, ModelBiasedLinearSearch_SIMD> }, \
    { {#L1, #L2, "gind", "model_biased_linear_simd"}, &experiment<key_type, rmi::RmiGInd<key_type, LT1, LT2>, ModelBiasedLinearSearch_SIMD> }, \
    { {#L1, #L2, "none", "model_biased_binary_branchless"}, &experiment<key_type, rmi::Rmi<key_type, LT1, LT2>, ModelBiasedBinarySearch_Branchless> }, \
    { {#L1, #L2, "labs", "model_biased_binary_branchless"}, &experiment<key_type, rmi::RmiLAbs<key_type, LT1, LT2>, ModelBiasedBinarySearch_Branchless> }, \
    { {#L1, #L2, "lind", "model_biased_binary_branchless"}, &experiment<key_type, rmi::RmiLInd<key_type, LT1, LT2>, ModelBiasedBinarySearch_Branchless> }, \
    { {#L1, #L2, "gabs", "model_biased_binary_branchless"}, &experiment<key_type, rmi::RmiGAbs<key_type, LT1, LT2>, ModelBiasedBinarySearch_Branchless> }, \
    { {#L1, #L2, "gind", "model_biased_binary_branchless"}, &experiment<key_type, rmi::RmiGInd<key_type, LT1, LT2>, ModelBiasedBinarySearch_Branchless> }, \
    
static std::map<Config, exp_fn_ptr, ConfigCompare> exp_map {
    //ENTRIES(linear_regression, linear_regression, rmi::LinearRegression, rmi::LinearRegression)
    //ENTRIES(linear_regression, linear_spline,     rmi::LinearRegression, rmi::LinearSpline)
    ENTRIES(linear_spline,     linear_regression, rmi::LinearSpline,     rmi::LinearRegression)
    ENTRIES(linear_spline,     linear_regression_welford, rmi::LinearSpline,     rmi::LinearRegression_welford)
    ENTRIES(linear_spline, linear_regression_welford_float, rmi::LinearSpline,     rmi::LinearRegression_float)
    //ENTRIES(linear_spline,     linear_spline,     rmi::LinearSpline,     rmi::LinearSpline)
    //ENTRIES(cubic_spline,      linear_regression, rmi::CubicSpline,      rmi::LinearRegression)
    //ENTRIES(cubic_spline,      linear_spline,     rmi::CubicSpline,      rmi::LinearSpline)
    //ENTRIES(radix,             linear_regression, rmi::Radix<key_type>,  rmi::LinearRegression)
    //ENTRIES(radix,             linear_spline,     rmi::Radix<key_type>,  rmi::LinearSpline)
}; ///< Map that assigns an experiment function pointer to RMI configurations.
#undef ENTRIES


/**
 * Triggers measurement of lookup times for an RMI configuration provided via command line arguments.
 * @param argc arguments counter
 * @param argv arguments vector
 */
int main(int argc, char *argv[])
{
    // Initialize argument parser.
    argparse::ArgumentParser program(argv[0], "0.1");

    // Define arguments.
    program.add_argument("filename")
        .help("path to binary file containing uin64_t keys");

    program.add_argument("layer1")
        .help("layer1 model type, either linear_regression, linear_spline, cubic_spline, or radix.");

    program.add_argument("layer2")
        .help("layer2 model type, either linear_regression, linear_spline, or cubic_spline.");

    program.add_argument("n_models")
        .help("number of models on layer2, power of two is recommended.")
        .action([](const std::string &s) { return std::stoul(s); });

    program.add_argument("bound_type")
        .help("type of error bounds used, either none, labs, lind, gabs, or gind.");

    program.add_argument("search")
        .help("search algorithm for error correction, either binary, model_biased_binary, exponential, model_biased_exponential, linear, or model_biased_linear.");

   program.add_argument("-n", "--n_reps")
        .help("number of experiment repetitions")
        .default_value(std::size_t(3))
        .action([](const std::string &s) { return std::stoul(s); });

    program.add_argument("-s", "--n_samples")
        .help("number of sampled lookup keys")
        .default_value(std::size_t(1'000'000))
        .action([](const std::string &s) { return std::stoul(s); });

    program.add_argument("switch_n")
        .help("number of keys that switch SIMD/SISD.")
        .action([](const std::string &s) { return std::stoul(s); });

    program.add_argument("--header")
        .help("output csv header")
        .default_value(false)
        .implicit_value(true);

    // Parse arguments.
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err) {
        std::cout << err.what() << '\n' << program;
        exit(EXIT_FAILURE);
    }

    // Read arguments.
    const auto filename = program.get<std::string>("filename");
    const auto dataset_name = split(filename, '/').back();
    const auto layer1 = program.get<std::string>("layer1");
    const auto layer2 = program.get<std::string>("layer2");
    const auto n_models = program.get<std::size_t>("n_models");
    const auto bound_type = program.get<std::string>("bound_type");
    const auto search = program.get<std::string>("search");
    const auto n_reps = program.get<std::size_t>("-n");
    const auto n_samples = program.get<std::size_t>("-s");
    const auto switch_n = program.get<std::size_t>("switch_n");

    // Load keys.
    auto keys = load_data<key_type>(filename);

    // Sample keys.
    uint64_t seed = 42;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distrib(0, keys.size() - 1);
    std::vector<key_type> samples;
    samples.reserve(n_samples);
    for (std::size_t i = 0; i != n_samples; ++i)
        samples.push_back(keys[distrib(gen)]);

    // Lookup experiment.
    Config config{layer1, layer2, bound_type, search};
    if (exp_map.find(config) == exp_map.end()) {
        std::cerr << "Error: " << layer1 << ',' << layer2 << ',' << bound_type << ',' << search << " is not a valid RMI configuration." << std::endl;
        exit(EXIT_FAILURE);
    }
    exp_fn_ptr exp_fn = exp_map[config];

    // Output header.
    if (program["--header"]  == true)
        std::cout << "dataset,"
                  << "n_keys,"
                  << "layer1,"
                  << "layer2,"
                  << "n_models,"
                  << "bounds,"
                  << "search,"
                  << "size_in_bytes,"
                  << "rep,"
                  << "n_samples,"
                  << "switch_n,"
                  << "lookup_time,"
                  << "lookup_accu,"
                  << std::endl;

    // Run experiment.
    (*exp_fn)(keys, n_models, samples, n_reps, dataset_name, layer1, layer2, bound_type, search, switch_n);

    exit(EXIT_SUCCESS);
}
