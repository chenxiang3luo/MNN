//
//  MobilenetUtils.cpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef ModelUtils_cpp
#define ModelUtils_cpp

#include "ModelUtils.hpp"
#include "Initializer.hpp"
#include <algorithm>

namespace MNN {
namespace Train {
namespace Model {

// https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
int makeDivisible(int v, int divisor, int minValue) {
    if (minValue == 0) {
        minValue = divisor;
    }
    int newV = std::max(minValue, int(v + divisor / 2) / divisor * divisor);

    // Make sure that round down does not go down by more than 10%.
    if (newV < 0.9 * v) {
        newV += divisor;
    }

    return newV;
}

std::shared_ptr<Module> DepthwiseSeparableConv2D(std::vector<int> inputOutputChannels, int stride) {
    return std::shared_ptr<Module>(new _DepthwiseSeparableConv2D(inputOutputChannels, stride));
}

std::shared_ptr<Module> ConvBnRelu(std::vector<int> inputOutputChannels, int kernelSize, int stride, bool depthwise, PaddingMode padmod) {
    return std::shared_ptr<Module>(new _ConvBnRelu(inputOutputChannels, kernelSize, stride, depthwise, padmod));
}

std::shared_ptr<Module> BottleNeck(std::vector<int> inputOutputChannels, int stride, int expandRatio) {
    return std::shared_ptr<Module>(new _BottleNeck(inputOutputChannels, stride, expandRatio));
}

std::shared_ptr<Module> Inception(int inputChannelSet, int channel_1x1, int channel_3x3_reduce, int channel_3x3,
                                  int channel_5x5_reduce, int channel_5x5, int channel_pool) {
    return std::shared_ptr<Module>(new _Inception(inputChannelSet, channel_1x1, channel_3x3_reduce, channel_3x3,
                                                      channel_5x5_reduce, channel_5x5, channel_pool));
}

std::shared_ptr<Module> FireMoudle(int inputChannel, int squeeze_1x1, int expand_1x1, int expand_3x3) {
    return std::shared_ptr<Module>(new _fireMoudle(inputChannel, squeeze_1x1, expand_1x1, expand_3x3));
}

std::shared_ptr<Module> IdentityBlock(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels) {
    return std::shared_ptr<Module>(new _IdentityBlock(kernelSize, inputChannels, outputChannels));
}

std::shared_ptr<Module> Resnet50ConvBlock(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels, int stride) {
    return std::shared_ptr<Module>(new _Resnet50ConvBlock(kernelSize, inputChannels, outputChannels, stride));
}

std::shared_ptr<Module> Resnet18BasicBlock(int inputChannel, int outputChannel, int kernelSize, int stride, bool convShortcut) {
    return std::shared_ptr<Module>(new _Resnet18BasicBlock(inputChannel, outputChannel, kernelSize, stride, convShortcut));
}

_DepthwiseSeparableConv2D::_DepthwiseSeparableConv2D(std::vector<int> inputOutputChannels, int stride) {
    int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];

    NN::ConvOption convOption;
    convOption.kernelSize = {3, 3};
    convOption.channel    = {inputChannels, inputChannels};
    convOption.padMode    = Express::SAME;
    convOption.stride     = {stride, stride};
    convOption.depthwise  = true;
    conv3x3.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

    bn1.reset(NN::BatchNorm(inputChannels));

    convOption.reset();
    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannels, outputChannels};
    convOption.padMode    = Express::SAME;
    convOption.stride     = {1, 1};
    convOption.depthwise  = false;
    conv1x1.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

    bn2.reset(NN::BatchNorm(outputChannels));

    registerModel({conv3x3, bn1, conv1x1, bn2});
}

std::vector<Express::VARP> _DepthwiseSeparableConv2D::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = conv3x3->forward(x);
    x = bn1->forward(x);
    x = _Relu6(x);
    x = conv1x1->forward(x);
    x = bn2->forward(x);
    x = _Relu6(x);

    return {x};
}

_ConvBnRelu::_ConvBnRelu(std::vector<int> inputOutputChannels, int kernelSize, int stride, bool depthwise, PaddingMode padmod) {
    int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];

    NN::ConvOption convOption;
    convOption.kernelSize = {kernelSize, kernelSize};
    convOption.channel    = {inputChannels, outputChannels};
    convOption.padMode    = padmod;
    convOption.stride     = {stride, stride};
    convOption.depthwise  = depthwise;
    conv.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

    bn.reset(NN::BatchNorm(outputChannels));

    registerModel({conv, bn});
}

std::vector<Express::VARP> _ConvBnRelu::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = conv->forward(x);
    x = bn->forward(x);
    x = _Relu6(x);

    return {x};
}

_BottleNeck::_BottleNeck(std::vector<int> inputOutputChannels, int stride, int expandRatio) {
    int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];
    int expandChannels = inputChannels * expandRatio;

    if (stride == 1 && inputChannels == outputChannels) {
        useShortcut = true;
    }

    if (expandRatio != 1) {
        layers.emplace_back(ConvBnRelu({inputChannels, expandChannels}, 1));
    }

    layers.emplace_back(ConvBnRelu({expandChannels, expandChannels}, 3, stride, true));

    NN::ConvOption convOption;
    convOption.kernelSize = {1, 1};
    convOption.channel    = {expandChannels, outputChannels};
    convOption.padMode    = Express::SAME;
    convOption.stride     = {1, 1};
    convOption.depthwise  = false;
    layers.emplace_back(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

    layers.emplace_back(NN::BatchNorm(outputChannels));

    registerModel(layers);
}

std::vector<Express::VARP> _BottleNeck::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    for (int i = 0; i < layers.size(); i++) {
        x = layers[i]->forward(x);
    }

    if (useShortcut) {
        x = x + inputs[0];
    }

    return {x};
}

_Inception::_Inception(int inputChannelSet, int channel_1x1,
                       int channel_3x3_reduce, int channel_3x3,
                       int channel_5x5_reduce, int channel_5x5,
                       int channel_pool) {
    NN::ConvOption convOption;
    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannelSet, channel_1x1};
    convOption.padMode    = Express::VALID;
    conv1.reset(NN::Conv(convOption));
    bn1.reset(NN::BatchNorm(channel_1x1));

    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannelSet, channel_3x3_reduce};
    convOption.padMode    = Express::VALID;
    conv2.reset(NN::Conv(convOption));
    bn2.reset(NN::BatchNorm(channel_3x3_reduce));

    convOption.kernelSize = {3, 3};
    convOption.channel    = {channel_3x3_reduce, channel_3x3};
    convOption.padMode    = Express::SAME;
    conv3.reset(NN::Conv(convOption));
    bn3.reset(NN::BatchNorm(channel_3x3));

    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannelSet, channel_5x5_reduce};
    convOption.padMode    = Express::VALID;
    conv4.reset(NN::Conv(convOption));
    bn4.reset(NN::BatchNorm(channel_5x5_reduce));

    convOption.kernelSize = {5, 5};
    convOption.channel    = {channel_5x5_reduce, channel_5x5};
    convOption.padMode    = Express::SAME;
    conv5.reset(NN::Conv(convOption));
    bn5.reset(NN::BatchNorm(channel_5x5));

    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannelSet, channel_pool};
    convOption.padMode    = Express::VALID;
    conv6.reset(NN::Conv(convOption));
    bn6.reset(NN::BatchNorm(channel_pool));

    convOption.kernelSize = {3, 3};
    convOption.stride = {1, 1};
    convOption.padMode    = Express::SAME;
    convOption.channel = {inputChannelSet, inputChannelSet};
    convOption.depthwise = true;
    conv_mp.reset(NN::Conv(convOption));

    registerModel({conv1, conv2, conv3, conv4, conv5, conv6,
                   bn1, bn2, bn3, bn4, bn5, bn6, conv_mp});
}

std::vector<Express::VARP> _Inception::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];
    auto inputChannel = x->getInfo()->dim[1];
    auto y1 = conv1->forward(x);
    y1=bn1->forward(y1);
    auto y2 = conv2->forward(x);
    y2=bn2->forward(y2);
    y2 = conv3->forward(y2);
    y2=bn3->forward(y2);
    auto y3 = conv4->forward(x);
    y3=bn4->forward(y3);
    y3 =conv5->forward(y3);
    y3=bn5->forward(y3);
    auto y4 = conv_mp->forward(x);
    y4 = conv6->forward(y4);
    y4 = bn6->forward(y4);
    auto z = _Concat({y1, y2, y3, y4}, 1);
    return {z};
}

_fireMoudle::_fireMoudle(int inputChannel, int squeeze_1x1, int expand_1x1, int expand_3x3) {
    NN::ConvOption convOption;
    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannel, squeeze_1x1};
    convOption.padMode    = Express::VALID;
    conv1.reset(NN::Conv(convOption));
    bn1.reset(NN::BatchNorm(squeeze_1x1));

    convOption.kernelSize = {1, 1};
    convOption.channel    = {squeeze_1x1, expand_1x1};
    convOption.padMode    = Express::VALID;
    conv2.reset(NN::Conv(convOption));
    bn2.reset(NN::BatchNorm(expand_1x1));

    convOption.kernelSize = {3, 3};
    convOption.channel    = {squeeze_1x1, expand_3x3};
    convOption.padMode    = Express::SAME;
    conv3.reset(NN::Conv(convOption));
    bn3.reset(NN::BatchNorm(expand_3x3));


    registerModel({conv1, conv2, conv3, bn1, bn2, bn3});
}

std::vector<Express::VARP> _fireMoudle::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x = conv1->forward(x);
    x = bn1->forward(x);
    auto y1 = conv2->forward(x);
    y1      = bn2->forward(y1);
    auto y2 = conv3->forward(x);
    y2      = bn3->forward(y2);
    auto z  = _Concat({y1, y2}, 1);
    return {z};
}

_IdentityBlock::_IdentityBlock(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels) {
    cbr1 = ConvBnRelu({inputChannels[0], outputChannels[0]}, 1, 1, false);
    cbr2 = ConvBnRelu({inputChannels[1], outputChannels[1]}, kernelSize, 1, false);

    NN::ConvOption convOption;
    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannels[2], outputChannels[2]};
    convOption.stride     = {1, 1};
    convOption.padMode    = SAME;
    conv.reset(NN::Conv(convOption));
    bn.reset(NN::BatchNorm(outputChannels[2]));
    registerModel({cbr1, cbr2, conv, bn});
}

std::vector<Express::VARP> _IdentityBlock::onForward(const std::vector<Express::VARP> &inputs) {
    VARP input_tensor = inputs[0];
    auto x = cbr1->forward(input_tensor);
    x      = cbr2->forward(x);
    x      = conv->forward(x);
    x      = bn->forward(x);
    x      = _Add(x, input_tensor);
    x      = _Relu(x);
    return {x};
}

_Resnet50ConvBlock::_Resnet50ConvBlock(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels, int stride) {
    cbr1 = ConvBnRelu({inputChannels[0], outputChannels[0]}, 1, stride);
    cbr2 = ConvBnRelu({inputChannels[1], outputChannels[1]}, kernelSize, 1);

    NN::ConvOption convOption;
    convOption.kernelSize = {1, 1};
    convOption.padMode    = SAME;

    convOption.channel    = {inputChannels[2], outputChannels[2]};
    convOption.stride     = {1, 1};
    conv.reset(NN::Conv(convOption));
    bn.reset(NN::BatchNorm(outputChannels[2]));

    convOption.channel    = {inputChannels[0], outputChannels[2]};
    convOption.stride     = {stride, stride};
    conv_sc.reset(NN::Conv(convOption));
    bn_sc.reset(NN::BatchNorm(outputChannels[2]));

    registerModel({cbr1, cbr2, conv, bn, conv_sc, bn_sc});
}

std::vector<Express::VARP> _Resnet50ConvBlock::onForward(const std::vector<Express::VARP> &inputs) {
    VARP input_tensor = inputs[0];
    auto x = cbr1->forward(input_tensor);
    x = cbr2->forward(x);
    x = conv->forward(x);
    x = bn->forward(x);

    auto shortcut = conv_sc->forward(input_tensor);
    shortcut = bn_sc->forward(shortcut);
    x = _Add(x, shortcut);
    x = _Relu(x);
    return {x};
}

_Resnet18BasicBlock::_Resnet18BasicBlock(int inputChannel, int outputChannel, int kernelSize, int stride, bool convShortcut) {
    NN::ConvOption convOption;
    convOption.kernelSize = {kernelSize, kernelSize};
    convOption.padMode    = SAME;

    convOption.channel    = {inputChannel, outputChannel};
    convOption.stride     = {stride, stride};

    conv1.reset(NN::Conv(convOption));
    bn1.reset(NN::BatchNorm(outputChannel));

    convOption.channel    = {outputChannel, outputChannel};
    convOption.stride     = {1, 1};
    conv2.reset(NN::Conv(convOption));
    bn2.reset(NN::BatchNorm(outputChannel));

    useConvShortcut = convShortcut;

    if (convShortcut) {
        convOption.channel    = {inputChannel, outputChannel};
        convOption.stride     = {stride, stride};

        conv_sc.reset(NN::Conv(convOption));
        bn_sc.reset(NN::BatchNorm(outputChannel));
        registerModel({conv1, conv2, bn1, bn2, conv_sc, bn_sc});
    } else {
        registerModel({conv1, conv2, bn1, bn2});
    }
}

std::vector<Express::VARP> _Resnet18BasicBlock::onForward(const std::vector<Express::VARP> &inputs) {
    VARP input_tensor = inputs[0];
    auto x = conv1->forward(input_tensor);
    x = bn1->forward(x);
    x = conv2->forward(x);
    x = bn2->forward(x);
    if (useConvShortcut) {
        auto sc = conv_sc->forward(input_tensor);
        sc = bn_sc->forward(sc);
        x = x + sc;
    } else {
        x = x + input_tensor;
    }
    return {x};
}

} // namespace Model
} // namespace Train
} // namespace MNN

#endif