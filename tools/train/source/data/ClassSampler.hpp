//
//  RandomSampler.hpp
//  MNN
//
//  Created by MNN on 2019/11/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ClassSampler_hpp
#define ClassSampler_hpp

#include <vector>
#include <map>
#include "Sampler.hpp"

namespace MNN {
namespace Train {

class MNN_PUBLIC ClassSampler : public Sampler {
public:
    explicit ClassSampler(std::vector<std::vector<size_t>> indices_class, bool shuffle);

    void reset(size_t size) override;
    void reset_class(uint8_t size);

    size_t size() override;

    const std::vector<size_t> indices();

    size_t index();

    std::vector<size_t> next(size_t batchSize) override;

private:
    std::vector<size_t> mIndices;
    std::vector<std::vector<size_t>> indice;
    size_t mIndex = 0;
    bool shuffle;
    bool mShuffle;
};

} // namespace Train
} // namespace MNN

#endif // RandomSampler
