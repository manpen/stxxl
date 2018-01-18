/***************************************************************************
 *  examples/containers/sorter2.cpp
 *
 *  Part of the STXXL. See http://stxxl.org
 *
 *  Copyright (C) 2013 Daniel Feist <daniel.feist@student.kit.edu>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *  (See accompanying file LICENSE_1_0.txt or copy at
 *  http://www.boost.org/LICENSE_1_0.txt)
 **************************************************************************/

#include <iostream>
#include <limits>

#include <tlx/logger.hpp>

#include <stxxl/random>
#include <stxxl/sorter>
#include <stxxl/timer>

struct TwoInteger
{
    int i, j;

    TwoInteger()
    { }

    TwoInteger(int _i, int _j)
        : i(_i), j(_j)
    { }
};

struct TwoIntegerComparator
{
    bool operator () (const TwoInteger& a, const TwoInteger& b) const
    {
        return a.i < b.i;
    }

    TwoInteger min_value() const
    {
        return TwoInteger(std::numeric_limits<int>::min(), std::numeric_limits<int>::min());
    }

    TwoInteger max_value() const
    {
        return TwoInteger(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
    }
};

int main()
{
    // template parameter <ValueType, CompareType, BlockSize(optional), AllocStr(optional)>
    using sorter_type = stxxl::sorter<TwoInteger, TwoIntegerComparator, 1*1024*1024>;

    // create sorter object (CompareType(), MainMemoryLimit)
    sorter_type int_sorter(TwoIntegerComparator(), 64 * 1024 * 1024);

    stxxl::random_number32 rand32;

    foxxll::timer Timer1;
    Timer1.start();

    // insert random numbers from [0,100000)
    for (size_t i = 0; i < 1000; ++i)
    {
        int_sorter.push(TwoInteger(rand32() % 100000, static_cast<int>(i)));    // fill sorter container
    }

    Timer1.stop();

    LOG1 << "push time: " << (Timer1.mseconds() / 1000);

    foxxll::timer Timer2;

    Timer2.start();
    int_sorter.sort();  // switch to output state and sort
    Timer2.stop();

    LOG1 << "sort time: " << (Timer2.mseconds() / 1000);

    // echo sorted elements
    while (!int_sorter.empty())
    {
        std::cout << int_sorter->i << " ";  // access value
        ++int_sorter;
    }

    return 0;
}
