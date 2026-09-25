#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "grad/utils/env.h"

namespace grad {

// A minimal persistent thread pool exposing one primitive:
//
//   parallel_for(n, grain, [](size_t begin, size_t end) { ... });
//
// The range [0, n) is split into chunks that workers (plus the calling
// thread) claim from an atomic counter, so uneven chunks self-balance.
// Falls back to running inline when n <= grain or the pool has one thread -
// spawning threads for tiny loops costs more than it saves. A grain of 0 is
// treated as 1.
//
// Nested calls (a parallel_for issued from inside a body, on a worker or on
// the calling thread) run inline: the outer loop already has every core
// busy, and the calling thread already holds the job lock, so trying to
// take it again would be undefined behavior rather than a clean refusal.
//
// The pool is created on first use with hardware_concurrency() threads
// (override with the GRAD_THREADS environment variable).
//
// Callers must ensure different indices touch disjoint data: the pool does
// no synchronization beyond the completion barrier at the end of each call.
//
// An exception thrown by fn on any thread is captured, the remaining chunks
// are abandoned, and it is rethrown on the calling thread once every worker
// has returned - so a throw inside a parallel body behaves like a throw from
// a plain loop rather than taking the process down with std::terminate.
class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    int threads() const { return num_threads_; }

    void parallel_for(size_t n, size_t grain, const std::function<void(size_t, size_t)>& fn) {
        if (n == 0) return;
        if (grain == 0) grain = 1;
        if (num_threads_ <= 1 || n <= grain || in_parallel_region()) {
            fn(0, n);
            return;
        }

        // Only one parallel job runs at a time. A job started by another
        // (non-pool) thread holds the lock; run this one inline rather
        // than wait, since that job already has every core busy.
        std::unique_lock<std::mutex> job_lock(job_mutex_, std::try_to_lock);
        if (!job_lock.owns_lock()) {
            fn(0, n);
            return;
        }

        chunk_size_ = grain;
        total_ = n;
        next_chunk_.store(0, std::memory_order_relaxed);
        body_ = &fn;

        {
            std::lock_guard<std::mutex> lk(wake_mutex_);
            job_id_++;
            workers_done_ = 0;
            first_error_ = nullptr;
        }
        wake_cv_.notify_all();

        work();  // the calling thread participates

        std::unique_lock<std::mutex> lk(wake_mutex_);
        done_cv_.wait(lk, [this] { return workers_done_ == num_threads_ - 1; });
        body_ = nullptr;
        if (first_error_) {
            std::rethrow_exception(std::exchange(first_error_, nullptr));
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(wake_mutex_);
            shutdown_ = true;
        }
        wake_cv_.notify_all();
        for (auto& t : threads_) t.join();
    }

private:
    ThreadPool() {
        int n = static_cast<int>(std::thread::hardware_concurrency());
        if (const char* value = env::lookup("GRAD_THREADS", "TRANSFORMER_THREADS")) {
            int v = std::atoi(value);
            if (v > 0) n = v;
        }
        num_threads_ = n > 0 ? n : 1;

        for (int i = 0; i < num_threads_ - 1; i++) {
            threads_.emplace_back([this] { worker_loop(); });
        }
    }

    void worker_loop() {
        uint64_t seen_job = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lk(wake_mutex_);
                wake_cv_.wait(lk, [&] { return shutdown_ || job_id_ != seen_job; });
                if (shutdown_) return;
                seen_job = job_id_;
            }
            work();
            {
                std::lock_guard<std::mutex> lk(wake_mutex_);
                workers_done_++;
            }
            done_cv_.notify_one();
        }
    }

    // True while this thread is executing a parallel_for body; read by
    // parallel_for to run nested calls inline.
    static bool& in_parallel_region() {
        static thread_local bool flag = false;
        return flag;
    }

    struct RegionGuard {
        bool previous;
        RegionGuard() : previous(std::exchange(in_parallel_region(), true)) {}
        ~RegionGuard() { in_parallel_region() = previous; }
        RegionGuard(const RegionGuard&) = delete;
        RegionGuard& operator=(const RegionGuard&) = delete;
    };

    void work() {
        const auto* fn = body_;
        if (!fn) return;
        RegionGuard region;
        while (true) {
            size_t begin = next_chunk_.fetch_add(chunk_size_, std::memory_order_relaxed);
            if (begin >= total_) break;
            size_t end = begin + chunk_size_;
            if (end > total_) end = total_;
            try {
                (*fn)(begin, end);
            } catch (...) {
                std::lock_guard<std::mutex> lk(wake_mutex_);
                if (!first_error_) first_error_ = std::current_exception();
                // Drain the range so every thread exits its loop promptly.
                next_chunk_.store(total_, std::memory_order_relaxed);
            }
        }
    }

    std::vector<std::thread> threads_;
    int num_threads_ = 1;

    std::mutex job_mutex_;  // serializes parallel_for calls
    std::mutex wake_mutex_;
    std::condition_variable wake_cv_;
    std::condition_variable done_cv_;
    uint64_t job_id_ = 0;
    int workers_done_ = 0;
    bool shutdown_ = false;
    std::exception_ptr first_error_;  // guarded by wake_mutex_

    const std::function<void(size_t, size_t)>* body_ = nullptr;
    std::atomic<size_t> next_chunk_{0};
    size_t chunk_size_ = 1;
    size_t total_ = 0;
};

// Convenience wrapper.
inline void parallel_for(size_t n, size_t grain, const std::function<void(size_t, size_t)>& fn) {
    ThreadPool::instance().parallel_for(n, grain, fn);
}

}  // namespace grad
