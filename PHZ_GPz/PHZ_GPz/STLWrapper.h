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

}  // namespace PHZ_GPz

#endif
