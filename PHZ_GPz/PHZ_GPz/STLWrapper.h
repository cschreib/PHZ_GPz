/**
 * @file PHZ_GPz/STLWrapper.h
 * @date 11/29/18
 * @author user
 *
 * @copyright (C) 2012-2020 Euclid Science Ground Segment
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 3.0 of the License, or (at your option)
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 *
 */

#ifndef _PHZ_GPZ_STL_WRAPPER_H
#define _PHZ_GPZ_STL_WRAPPER_H

#include <algorithm>
#include <memory>
#include <vector>
#include <numeric>
#include <chrono>
#include <thread>
#include <mutex>
#include <cassert>

namespace PHZ_GPz {

    using uint_t = std::size_t;

    using histogram_iterator = std::vector<uint_t>::const_iterator;

    template<typename T, typename B, typename F>
    void histogram(const T& data, const B& bins, F&& func) {
        std::vector<uint_t> ids(data.size());
        std::iota(ids.begin(), ids.end(), 0u);

        uint_t nbin = bins.size() - 1;
        auto first = ids.begin();
        for (uint_t i = 0; i < nbin; ++i) {
            auto last = std::partition(first, ids.end(), [&bins,&data,i](uint_t id) {
                return data[id] >= bins[i] && data[id] < bins[i+1];
            });

            func(i, first, last);

            if (last == ids.end()) break;
        }
    }

    template<typename T>
    T square(T data) {
        return data*data;
    }

    inline double now() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count()*1e-6;
    }

    struct parallel_for {
        uint_t chunk_size = 0;

    private :

        struct worker_t {
            std::unique_ptr<std::thread> thread;
            parallel_for& pfor;

            explicit worker_t(parallel_for& p) : pfor(p) {}

            worker_t(worker_t&& w) : pfor(w.pfor) {}

            worker_t(const worker_t& w) = delete;

            template<typename F>
            void start(const F& f) {
                thread = std::unique_ptr<std::thread>(new std::thread([this,&f]() {
                    uint_t i0, i1;
                    while (pfor.query_chunk(i0, i1)) {
                        for (uint_t i = i0; i < i1; ++i) {
                            f(i);
                        }
                    }
                }));
            }

            void join() {
                thread->join();
                thread = nullptr;
            }
        };

        // Workers
        std::vector<parallel_for::worker_t> workers;

        // Internal
        std::mutex query_mutex;
        uint_t n, i0, i1, di;

        bool query_chunk(uint_t& oi0, uint_t& oi1) {
            std::unique_lock<std::mutex> l(query_mutex);

            if (i0 == n) {
                return false;
            }

            oi0 = i0;
            oi1 = i1;

            i0 = i1;
            i1 += di;
            if (i1 > n) {
                i1 = n;
            }

            return true;
        }

    public :

        parallel_for() = default;
        parallel_for(const parallel_for&) = delete;
        parallel_for(parallel_for&&) = delete;

        explicit parallel_for(uint_t nthread) {
            for (uint_t i = 0; i < nthread; ++i) {
                workers.emplace_back(*this);
            }
        }

        template<typename F>
        void execute(const F& f, uint_t ifirst, uint_t ilast) {
            assert(ilast >= ifirst && "loop index must be increasing");

            n = ilast - ifirst;

            if (workers.empty()) {
                // Single-threaded execution
                for (uint_t i = ifirst; i < ilast; ++i) {
                    f(i);
                }
            } else {
                // Multi-threaded execution
                uint_t nchunk = (chunk_size == 0 ?
                    workers.size() : std::max(workers.size(), n/chunk_size));

                // Setup chunks
                di = n/nchunk + 1;
                i0 = ifirst;
                i1 = ifirst + di;

                // Launch threads
                for (auto& t : workers) {
                    t.start(f);
                }

                // Join all
                for (auto& t : workers) {
                    t.join();
                }
            }
        }

        template<typename F>
        void execute(const F& f, uint_t i1) {
            execute(f, 0, i1);
        }

        uint_t size() const {
            return workers.size();
        }
    };


}  // namespace PHZ_GPz

#endif
