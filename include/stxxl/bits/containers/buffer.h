/***************************************************************************
 *  include/stxxl/bits/containers/em_buffer.h
 *
 *  Part of the STXXL. See http://stxxl.sourceforge.net
 *
 *  Copyright (C) 2017 Manuel Penschuck <stxxl@manuel.jetzt>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *  (See accompanying file LICENSE_1_0.txt or copy at
 *  http://www.boost.org/LICENSE_1_0.txt)
 **************************************************************************/

#ifndef STXXL_CONTAINERS_EM_BUFFER_HEADER
#define STXXL_CONTAINERS_EM_BUFFER_HEADER

#include <cassert>
#include <atomic>
#include <limits>
#include <vector>
#include <mutex>

#include <tlx/define.hpp>
#include <tlx/die.hpp>
#include <tlx/simple_vector.hpp>

#include <foxxll/mng/block_manager.hpp>
#include <foxxll/mng/write_pool.hpp>
#include <foxxll/mng/typed_block.hpp>

#include <stxxl/bits/defines.h>
#include <stxxl/bits/common/padding.h>

namespace stxxl {
/**
 * The EM buffer can be used to store elements without any guarantees
 * regarding their order. Similarly to the sorter, the EM buffer
 * has two modes: element-wise pushes, followed (but not interleaved)
 * by a data extraction phase. It has a thread-safe push interface, which
 * minimizes the synchronization with other threads (which may only
 * occur at block boundaries). Data can be extracted using a bulk_pull
 * interface and is thread-safe as well. However, each bulk_pull may block
 * until data is read from disk.
 *
 * \code
 * em_buffer<data_type> buf(16);
 *
 * #pragma omp parallel
 * {
 *  const auto tid = omp_get_num_thread();
 *  #pragma omp for
 *  for( ... ) {
 *    data_type value = // ...
 *    buf.push(value, tid);
 *  }
 *  buf.finished_pushing(tid);
 * }
 *
 * while(buf.size()) {
 *  std::vector<data_type> res = buf.bulk_pull();
 * }
 * \endcode
 *
 * @tparam ValueType
 * @tparam BlockSize
 * @tparam AllocStr
 */

template <
    typename ValueType,
    size_t BlockSize = STXXL_DEFAULT_BLOCK_SIZE(ValueType),
    class AllocStr = foxxll::default_alloc_strategy
>
class buffer {
    static constexpr bool debug = true;

public:
    using value_type = ValueType;
    static constexpr size_t block_size = BlockSize;
    using alloc_strategy_type = AllocStr;
    using thread_index = uint32_t;

    using block_type = foxxll::typed_block<block_size, value_type>;
    using bid_type = foxxll::BID<block_size>;
    using pool_type = foxxll::write_pool<block_type>;

    buffer(thread_index num_threads, pool_type* pool) :
        num_threads_(num_threads),
        num_active_threads_(num_threads),
        bm_(foxxll::block_manager::get_ref()),
        pool_(pool),
        destroy_pool_(false),
        cursors_(num_threads)
    {}

    explicit buffer(thread_index num_threads, int blocks = -1) :
        buffer(num_threads, nullptr)
    {
        LOG << "buffer[" << this << "] created. "
            "num_threads=" << num_threads << ", "
            "block_size=" << size_t(block_size) << "bytes / " << size_t(block_type::size) << "items.";

        if (blocks < 0) blocks = static_cast<int>(1.5 * num_threads);
        assert(static_cast<thread_index>(blocks) >= num_threads);

        pool_ = new pool_type(blocks);
        destroy_pool_ = true;
    }

    ~buffer() {
        for(auto& cur : cursors_) {
            if (cur.block != nullptr) {
                pool_->add(cur.block);
            }
        }

        if (destroy_pool_) {
            assert(pool_);
            delete(pool_);
        }
    }

    //! Copy element v into the insertion buffer of thread tid.
    void push(const value_type& v, thread_index tid) {
        assert(mode_.load() == Mode::Push);
        assert(tid < num_threads_);
        cursor_type& cur = cursors_[tid];
        assert(!cur.done);

        if (TLX_UNLIKELY(cur.block == nullptr)) {
            // allocate block if we do not currently have one
            std::unique_lock<std::mutex> lock(mutex_pool_);
            cur.block = pool_->steal();
        }

        // assert there's still space in our current block and
        // write element to it
        assert(cur.elements < block_type::size);
        cur.block->elem[cur.elements++] = v;

        // write out block to EM if it is full
        if (TLX_UNLIKELY(cur.elements == block_type::size)) {
            write_block_(cur);
        }
    }

    //! Indicate that thread tid will not push any more.
    //! Has to be called by all threads before bulk_pull becomes available.
    void finished_pushing(thread_index tid) {
        LOG << "buffer[" << this << "] finished pushing with thread " << tid;
        assert(mode_.load() == Mode::Push);

        // check that this thread did not already finish and then mark it
        cursor_type &cur = cursors_[tid];
        assert(!cur.done);

        // try to merge remaining elements with other incomplete buffers
        std::unique_lock<std::mutex> lock(mutex_merge_);
        if (cur.elements != 0) {
            // TODO: In some cases it may be better to append the other
            // data to our block (less copying)

            for(size_t i=0; i<cursors_.size(); i++) {
                auto& other = cursors_[i];
                if (!other.done || other.block == nullptr)
                    continue;

                // copy as many elements to the other buffer as possible
                const auto capacity = std::min<size_t>(
                    cur.elements,
                    block_type::size - other.elements);

                std::copy_n(&cur.block->elem[cur.elements - capacity], capacity,
                    &other.block->elem[other.elements]);

                LOG << "Copy " << capacity << " of " << cur.elements << " items from " << tid << "'s buffer to " << i << "'s with currently " << other.elements << " items";

                // and it other block is now filled, write it to disk
                other.elements += capacity;
                cur.elements -= capacity;

                if (other.elements == block_type::size) {
                    write_block_(other);
                }

                if (!cur.elements)
                    break;
            }
        }

        cur.done = true;

        // give block back to pool, if we can
        if (cur.elements == 0 && cur.block) {
            std::unique_lock<std::mutex> lock(mutex_pool_);
            pool_->add(cur.block);
        }
        lock.unlock();

        if (! --num_active_threads_) {
            for(size_t i=0; i < cursors_.size(); i++)
                LOG << "i=" << i << " " << cursors_[i].elements << " @ " << cursors_[i].block;

            // this is the last thread

            size_t remaining_cursor = 0;
            for(size_t i=0; i < cursors_.size(); i++) {
                if (cursors_[i].block != nullptr) {
                    remaining_cursor = i;
                    break;
                }
            }

            #ifndef NDEBUG
                // make sure we properly returned all blocks
                for (thread_index t = 0; t < num_threads_; t++) {
                    die_unless(cursors_[t].block == nullptr || t == remaining_cursor);
                    die_unless(cursors_[t].done);
                }
            #endif

            // make sure to move the last remaining non-empty block to the first slot
            // (for faster pulling)
            if (cursors_[remaining_cursor].elements && remaining_cursor != 0) {
                cursors_[0] = cursors_[remaining_cursor];
                cursors_[remaining_cursor].block = nullptr; // ownership was transferred to cursor 0
            }

            if (cursors_[0].block == nullptr) {
                cursors_[0].elements = 0;
            }

            // From now on, the user is allows to use (only) the Pull interface
            size_.store(bids_.size() * block_type::size + cursors_[0].elements);
            mode_.store(Mode::Pull);
        }
    }

    //! Extract upto max_size elements from buffer. In case max_size is large than
    //! size() all elements are retrieved.
    //! \warning A call is only valid if all threads called finished_pushing()
    std::vector<value_type> pull(size_t max_size = std::numeric_limits<size_t>::max()) {
        assert(mode_.load() == Mode::Pull);
        std::vector<value_type> result;

        size_t offset = 0; // first element in result not loaded yet

        std::unique_lock<std::mutex> lock(mutex_bids_);

        // In case the buffer is empty stop here
        if (!size()) return result;

        const auto max_blocks = std::min<size_t>(bids_.size(), max_size / block_type::size);
        const auto elems_to_fetch = std::min<size_t>(max_size, max_blocks * block_type::size + cursors_[0].elements);

        // TODO: avoid default construction
        result.resize(elems_to_fetch);

        // prepare to collect requests
        std::vector<foxxll::request_ptr> requests;
        requests.reserve(max_blocks);

        // issue all read requests
        const size_t remaining_bids = bids_.size() - max_blocks;
        for(auto it = std::next(bids_.begin(), remaining_bids); it != bids_.end(); ++it, offset += block_type::size) {
            requests.push_back(it->read(&result[offset], block_size));
        }

        bids_.resize(remaining_bids);
        bids_.shrink_to_fit();

        if (offset < max_size && cursors_[0].elements) {
            const size_t elements_to_copy = std::min(cursors_[0].elements, max_size - offset);
            const size_t first_elem = cursors_[0].elements - elements_to_copy;

            assert(cursors_[0].block);
            std::copy_n(&cursors_[0].block->elem[first_elem], elements_to_copy, &result[offset]);

            offset += elements_to_copy;
            cursors_[0].elements -= elements_to_copy;

            if (!cursors_[0].elements) {
                std::unique_lock<std::mutex> lock(mutex_pool_);
                pool_->add(cursors_[0].block);
            }
        }

        assert(size() >= offset);
        size_ -= offset;

        lock.unlock();

        foxxll::wait_all(requests.begin(), requests.end());

        return result;
    }

    //! Returns the number of items that are still stored in the buffer
    //! \warning A call is only valid if all threads called finished_pushing()
    size_t size() const {
        assert(mode_.load() == Mode::Pull);
        return size_;
    }

private:
    template <size_t Padding>
    struct bid_with_size {
        bid_type bid;
        size_t elements;
    };

    static constexpr size_t cache_line_width = 64; // bytes

    struct cursor_type : private stxxl::padding<
        static_cast<size_t>( (-1 * static_cast<ptrdiff_t>(sizeof(bid_type) + 2*sizeof(size_t))) % cache_line_width )
    > {
        size_t elements;
        block_type* block;
        std::atomic<bool> done;

        cursor_type() : elements(0), block(nullptr), done(false)
        {}

        ~cursor_type() {
            assert(block == nullptr);
        };

        cursor_type& operator=(const cursor_type& o) {
            elements = o.elements;
            block = o.block;
            done.store(o.done.load());
            return *this;
        }
    };

    enum class Mode {Push, Pull};

    //! State the container is in -- either pushing (initial) or pulling (final)
    std::atomic<Mode> mode_ {Mode::Push};

    //! Number of threads supported by this container
    size_t num_threads_;

    //! Number of threads that did not yet call finished_pushing
    std::atomic<size_t> num_active_threads_;

    //! Reference to the global block manager
    foxxll::block_manager& bm_;

    //! Pool is only used during pushing -- for pulling we allocate memory via new()
    pool_type* pool_;

    //! If pool is constructed by this class it will be destroyed automatically
    bool destroy_pool_;

    alloc_strategy_type alloc_strategy_;

    //! This mutex is used during pushing phase to avoid data races on bids_
    std::mutex mutex_bids_;

    //! This mutex is used during finish_pushing to avoid races on cursors_
    std::mutex mutex_merge_;

    //! This mutex locks the pool
    std::mutex mutex_pool_;

    //! Storage for all bids written to
    std::vector<bid_type> bids_;

    //! State of pushing threads
    tlx::SimpleVector<cursor_type> cursors_;

    //! Number of elements in buffer -- only valid during pulling
    std::atomic<size_t> size_;

    void write_block_(cursor_type& cur) {
        bid_type new_bid;

        // allocate memory on disk and write block to disk
        bm_.new_block(alloc_strategy_, new_bid, bids_.size());
        LOG0 << "buffer[" << this << "] write block " << cur.block << " to bid " << new_bid;

        {
            std::unique_lock<std::mutex> lock(mutex_pool_);
            pool_->write(cur.block, new_bid);
        }
        cur.elements = 0;

        // store bid to later read the data again
        {
            std::unique_lock<std::mutex> lock(mutex_bids_);
            bids_.push_back(new_bid);
        }
    }
};


} // namespace stxxl

#endif // STXXL_CONTAINERS_EM_BUFFER_HEADER
