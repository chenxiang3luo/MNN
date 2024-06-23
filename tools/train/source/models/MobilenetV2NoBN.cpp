//
//  MobilenetV2NoBN.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef MobilenetV2NoBN_cpp
#define MobilenetV2NoBN_cpp

#include <algorithm>
#include "MobilenetV2NoBN.hpp"
#include <iostream>
namespace MNN {
namespace Train {
namespace Model {
using namespace MNN::Express;

MobilenetV2NoBN::MobilenetV2NoBN(int numClasses, float widthMult, int divisor) {
    int inputChannels = 32;
    int lastChannels  = 1280;

    std::vector<std::vector<int> > invertedResidualSetting;
    invertedResidualSetting.push_back({1, 16, 1, 1});
    invertedResidualSetting.push_back({6, 24, 2, 2});
    invertedResidualSetting.push_back({6, 32, 3, 2});
    invertedResidualSetting.push_back({6, 64, 4, 2});
    invertedResidualSetting.push_back({6, 96, 3, 1});
    invertedResidualSetting.push_back({6, 160, 3, 2});
    invertedResidualSetting.push_back({6, 320, 1, 1});

    inputChannels = makeDivisible(inputChannels * widthMult, divisor);
    lastChannels  = makeDivisible(lastChannels * std::max(1.0f, widthMult), divisor);

    firstConv = ConvBnReluNoBN({3, inputChannels}, 3, 2);

    for (int i = 0; i < invertedResidualSetting.size(); i++) {
        std::vector<int> setting = invertedResidualSetting[i];
        int t                    = setting[0];
        int c                    = setting[1];
        int n                    = setting[2];
        int s                    = setting[3];

        int outputChannels = makeDivisible(c * widthMult, divisor);

        for (int j = 0; j < n; j++) {
            int stride = 1;
            if (j == 0) {
                stride = s;
            }

            bottleNeckBlocks.emplace_back(BottleNeckNoBN({inputChannels, outputChannels}, stride, t));
            inputChannels = outputChannels;
        }
    }

    lastConv = ConvBnReluNoBN({inputChannels, lastChannels}, 1);

    dropout.reset(NN::Dropout(0.1));
    fc.reset(NN::Linear(lastChannels, numClasses, true, std::shared_ptr<Initializer>(Initializer::MSRA())));

    NN::ConvOption convOption;
    convOption.channel = {lastChannels, lastChannels};
    convOption.kernelSize = {7, 7};
    convOption.depthwise = true;
    convOption.padMode = SAME;
    convOption.stride = {7, 7};
    conv_ap.reset(NN::Conv(convOption));

    registerModel({firstConv, lastConv, dropout, fc, conv_ap});
    registerModel(bottleNeckBlocks);
    setName("MobilenetV2NoBN");
}

std::vector<Express::VARP> MobilenetV2NoBN::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = firstConv->forward(x);

    for (int i = 0; i < bottleNeckBlocks.size(); i++) {
        x = bottleNeckBlocks[i]->forward(x);
    }

    x = lastConv->forward(x);

    // global avg pooling
//    x->getInfo()->printShape();
    x = conv_ap->forward(x);

    x = _Convert(x, NCHW);
    x = _Reshape(x, {0, -1});

    x = dropout->forward(x);
    x = fc->forward(x);

    x = _Softmax(x, 1);
    return {x};
}

std::vector<Express::VARP> MobilenetV2NoBN::onEmbedding(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = firstConv->forward(x);

    for (int i = 0; i < bottleNeckBlocks.size(); i++) {
        x = bottleNeckBlocks[i]->forward(x);
    }

    x = lastConv->forward(x);

    // global avg pooling
//    x->getInfo()->printShape();
    x = conv_ap->forward(x);

    x = _Convert(x, NCHW);
    x = _Reshape(x, {0, -1});
    
    return {x};
}

Express::VARP MobilenetV2NoBN::embedding(Express::VARP input) {
    return this->onEmbedding({input})[0];
}

} // namespace Model
} // namespace Train
} // namespace MNN
#endif