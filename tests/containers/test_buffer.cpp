#include <omp.h>

#include <array>
#include <iostream>
#include <sstream>
#include <random>

#include <tlx/logger.hpp>
#include <tlx/die.hpp>
#include <tlx/string.hpp>

#include <stxxl/seed>
#include <stxxl/bits/containers/buffer.h>

constexpr size_t block_size = 128;
using data_type = uint32_t;

template<size_t N=3>
class checksum {
public:
    explicit checksum() {
        sum_.fill(0);
    }

    void push(uint64_t value) {
        uint64_t tmp = 1;
        for(size_t i=0; i<N; i++) {
            tmp *= value;
            sum_[i] += tmp;
        }
        size_++;
    }

    bool operator==(const checksum& o) const {
        return size_ == o.size_ && sum_ == o.sum_;
    }

    checksum& operator+=(const checksum& o) {
        size_ += o.size_;
        for(size_t i=0; i<N; i++)
            sum_[i] += o.sum_[i];
        return *this;
    }

    friend std::ostream& operator<< (std::ostream&o, const checksum& cs) {
        o << ("[n=" + std::to_string(cs.size_) + ", "
              "cs=" + tlx::join(", ", cs.sum_.cbegin(), cs.sum_.cend()) + "]");
        return o;
    }

    size_t size() const {return size_;}

private:
    using cs_type = std::array<uint64_t, N>;

    cs_type sum_;
    size_t size_ {0};
};

void multithreaded_push_pull(const size_t n, const int threads, const size_t variance = 0) {
    constexpr bool debug = true;
    checksum<> cs_global;

    using buffer_type = stxxl::buffer<data_type, block_size>;
    buffer_type buffer(threads);

    const size_t expected_num = n * threads + variance * (threads - 1) * (threads) / 2;

    {
        foxxll::scoped_print_iostats stats(
            "Push n=" + std::to_string(expected_num),
            expected_num * sizeof(buffer_type::value_type));

        #pragma omp parallel num_threads(threads)
        {
            const auto tid = omp_get_thread_num();
            checksum<> cs;
            std::mt19937 randgen(stxxl::seed_sequence::get_ref().get_next_seed());
            std::uniform_int_distribution<uint32_t> distr;

            for (size_t i = 0; i < n + variance * tid; i++) {
                const auto value = distr(randgen);
                cs.push(value);
                buffer.push(value, tid);
                LOG0 << "push i=" << i << " value=" << value;
            }

            buffer.finished_pushing(tid);

            #pragma omp critical
            cs_global += cs;
        };
    }

    die_unequal(expected_num, cs_global.size());
    die_unequal(expected_num, buffer.size());

    checksum<> cs_global_pull;
    {
        foxxll::scoped_print_iostats stats(
            "Pull n=" + std::to_string(expected_num),
            expected_num * sizeof(buffer_type::value_type));

        #pragma omp parallel num_threads(threads)
        {
            checksum<> cs;

            while (1) {
                const std::vector<uint32_t> result(buffer.pull(1024));

                if (result.empty()) break;

                size_t i = 0;
                for (const auto x : result) {
                    cs.push(x);
                    LOG0 << "pull i=" << i++ << " value=" << x;
                }
            }

            #pragma omp critical
            cs_global_pull += cs;
        }
    }

    die_unequal(cs_global, cs_global_pull);
}

int main() {
    std::mt19937 randgen;

    constexpr size_t elem_per_block = block_size / sizeof(data_type);

    // single threaded around the block size
    multithreaded_push_pull(elem_per_block-1, 1, 0);
    multithreaded_push_pull(elem_per_block, 1, 0);
    multithreaded_push_pull(elem_per_block+1, 1, 0);
    multithreaded_push_pull(2*elem_per_block-1, 1, 0);
    multithreaded_push_pull(2*elem_per_block, 1, 0);
    multithreaded_push_pull(2*elem_per_block+1, 1, 0);


    const auto max_threads = omp_get_max_threads();

    std::uniform_int_distribution<size_t> distr(10, 100000);
    for(int i=0; i < 100; i++) {
        multithreaded_push_pull(distr(randgen), max_threads, distr(randgen)/100);
    }

    return 0;
}
