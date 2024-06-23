//
//  RandomSampler.cpp
//  MNN
//
//  Created by MNN on 2019/11/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ClassSampler.hpp"
#include <algorithm>
#include <iterator>
#include <random>
#include "Distributions.hpp"
#include "RandomGenerator.hpp"
using namespace MNN::Express;

namespace MNN {
namespace Train {

ClassSampler::ClassSampler(std::vector<std::vector<size_t>> indices_class, bool shuffle) {
    // mIndices.reserve(size);
    // for (int i = 0; i < size; i++) {
    //     mIndices.emplace_back(i);
    // }
    mShuffle = shuffle;
    indice = std::move(indices_class);

    // mShuffle = shuffle;
    // if (mShuffle) {
    //     std::shuffle(mIndices.begin(), mIndices.end(), RandomGenerator::generator());
    // }
}

void ClassSampler::reset(size_t size) {
    mIndices.clear();
    mIndices.reserve(size);
    for (int i = 0; i < size; i++) {
        mIndices.emplace_back(i);
    }

    if (mShuffle) {
        std::shuffle(mIndices.begin(), mIndices.end(), RandomGenerator::generator());
    }

    mIndex = 0;
}

void ClassSampler::reset_class(uint8_t size){
    mIndices = indice[size];
    if (mShuffle) {
        std::shuffle(mIndices.begin(), mIndices.end(), RandomGenerator::generator());
    }
}

size_t ClassSampler::size() {
    return mIndices.size();
}

const std::vector<size_t> ClassSampler::indices() {
    return mIndices;
}

size_t ClassSampler::index() {
    return mIndex;
}

std::vector<size_t> ClassSampler::next(size_t batchSize) {
    MNN_ASSERT(mIndex <= mIndices.size());

    auto remainIndices = mIndices.size();
    if (remainIndices == 0) {
        return {};
    }

    std::vector<size_t> batchIndex(batchSize);
    std::copy(mIndices.begin(), mIndices.begin() + batchIndex.size(), batchIndex.begin());

    return batchIndex;
}

} // namespace Train
} // namespace MNN
