//
//  MobilenetUtils.cpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef ModelUtilsNoBN_cpp
#define ModelUtilsNoBN_cpp

#include "ModelUtilsNoBN.hpp"
#include "Initializer.hpp"
#include <algorithm>

namespace MNN {
namespace Train {
namespace Model {

std::shared_ptr<Module> DepthwiseSeparableConv2DNoBN(std::vector<int> inputOutputChannels, int stride) {
    return std::shared_ptr<Module>(new _DepthwiseSeparableConv2DNoBN(inputOutputChannels, stride));
}

std::shared_ptr<Module> ConvBnReluNoBN(std::vector<int> inputOutputChannels, int kernelSize, int stride, bool depthwise, PaddingMode padmod) {
    return std::shared_ptr<Module>(new _ConvBnReluNoBN(inputOutputChannels, kernelSize, stride, depthwise, padmod));
}

std::shared_ptr<Module> BottleNeckNoBN(std::vector<int> inputOutputChannels, int stride, int expandRatio) {
    return std::shared_ptr<Module>(new _BottleNeckNoBN(inputOutputChannels, stride, expandRatio));
}

std::shared_ptr<Module> InceptionNoBN(int inputChannelSet, int channel_1x1, int channel_3x3_reduce, int channel_3x3,
                                  int channel_5x5_reduce, int channel_5x5, int channel_pool) {
    return std::shared_ptr<Module>(new _InceptionNoBN(inputChannelSet, channel_1x1, channel_3x3_reduce, channel_3x3,
                                                      channel_5x5_reduce, channel_5x5, channel_pool));
}

std::shared_ptr<Module> FireMoudleNoBN(int inputChannel, int squeeze_1x1, int expand_1x1, int expand_3x3) {
    return std::shared_ptr<Module>(new _fireMoudleNoBN(inputChannel, squeeze_1x1, expand_1x1, expand_3x3));
}

std::shared_ptr<Module> IdentityBlockNoBN(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels) {
    return std::shared_ptr<Module>(new _IdentityBlockNoBN(kernelSize, inputChannels, outputChannels));
}

std::shared_ptr<Module> Resnet50ConvBlockNoBN(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels, int stride) {
    return std::shared_ptr<Module>(new _Resnet50ConvBlockNoBN(kernelSize, inputChannels, outputChannels, stride));
}

std::shared_ptr<Module> Resnet18BasicBlockNoBN(int inputChannel, int outputChannel, int kernelSize, int stride, bool convShortcut) {
    return std::shared_ptr<Module>(new _Resnet18BasicBlockNoBN(inputChannel, outputChannel, kernelSize, stride, convShortcut));
}

_DepthwiseSeparableConv2DNoBN::_DepthwiseSeparableConv2DNoBN(std::vector<int> inputOutputChannels, int stride) {
    int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];

    NN::ConvOption convOption;
    convOption.kernelSize = {3, 3};
    convOption.channel    = {inputChannels, inputChannels};
    convOption.padMode    = Express::SAME;
    convOption.stride     = {stride, stride};
    convOption.depthwise  = true;
    conv3x3.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

    convOption.reset();
    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannels, outputChannels};
    convOption.padMode    = Express::SAME;
    convOption.stride     = {1, 1};
    convOption.depthwise  = false;
    conv1x1.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

    registerModel({conv3x3, conv1x1});
}

std::vector<Express::VARP> _DepthwiseSeparableConv2DNoBN::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = conv3x3->forward(x);
    x = _Relu6(x);
    x = conv1x1->forward(x);
    x = _Relu6(x);

    return {x};
}

_ConvBnReluNoBN::_ConvBnReluNoBN(std::vector<int> inputOutputChannels, int kernelSize, int stride, bool depthwise, PaddingMode padmod) {
    int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];

    NN::ConvOption convOption;
    convOption.kernelSize = {kernelSize, kernelSize};
    convOption.channel    = {inputChannels, outputChannels};
    convOption.padMode    = padmod;
    convOption.stride     = {stride, stride};
    convOption.depthwise  = depthwise;
    conv.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));


    registerModel({conv});
}

std::vector<Express::VARP> _ConvBnReluNoBN::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = conv->forward(x);
    x = _Relu6(x);

    return {x};
}

_BottleNeckNoBN::_BottleNeckNoBN(std::vector<int> inputOutputChannels, int stride, int expandRatio) {
    int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];
    int expandChannels = inputChannels * expandRatio;

    if (stride == 1 && inputChannels == outputChannels) {
        useShortcut = true;
    }

    if (expandRatio != 1) {
        layers.emplace_back(ConvBnReluNoBN({inputChannels, expandChannels}, 1));
    }

    layers.emplace_back(ConvBnReluNoBN({expandChannels, expandChannels}, 3, stride, true));

    NN::ConvOption convOption;
    convOption.kernelSize = {1, 1};
    convOption.channel    = {expandChannels, outputChannels};
    convOption.padMode    = Express::SAME;
    convOption.stride     = {1, 1};
    convOption.depthwise  = false;
    layers.emplace_back(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

    registerModel(layers);
}

std::vector<Express::VARP> _BottleNeckNoBN::onForward(const std::vector<Express::VARP> &inputs) {
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

_InceptionNoBN::_InceptionNoBN(int inputChannelSet, int channel_1x1,
                               int channel_3x3_reduce, int channel_3x3,
                               int channel_5x5_reduce, int channel_5x5,
                               int channel_pool) {
    NN::ConvOption convOption;
    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannelSet, channel_1x1};
    convOption.padMode    = Express::VALID;
    conv1.reset(NN::Conv(convOption));

    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannelSet, channel_3x3_reduce};
    convOption.padMode    = Express::VALID;
    conv2.reset(NN::Conv(convOption));

    convOption.kernelSize = {3, 3};
    convOption.channel    = {channel_3x3_reduce, channel_3x3};
    convOption.padMode    = Express::SAME;
    conv3.reset(NN::Conv(convOption));

    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannelSet, channel_5x5_reduce};
    convOption.padMode    = Express::VALID;
    conv4.reset(NN::Conv(convOption));

    convOption.kernelSize = {5, 5};
    convOption.channel    = {channel_5x5_reduce, channel_5x5};
    convOption.padMode    = Express::SAME;
    conv5.reset(NN::Conv(convOption));

    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannelSet, channel_pool};
    convOption.padMode    = Express::VALID;
    conv6.reset(NN::Conv(convOption));

    convOption.kernelSize = {3, 3};
    convOption.stride = {1, 1};
    convOption.padMode    = Express::SAME;
    convOption.channel = {inputChannelSet, inputChannelSet};
    convOption.depthwise = true;
    conv_mp.reset(NN::Conv(convOption));

    registerModel({conv1, conv2, conv3, conv4, conv5, conv6, conv_mp});
}

std::vector<Express::VARP> _InceptionNoBN::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];
    auto inputChannel = x->getInfo()->dim[1];
    auto y1 = conv1->forward(x);
    auto y2 = conv2->forward(x);
    y2 = conv3->forward(y2);
    auto y3 = conv4->forward(x);
    y3 =conv5->forward(y3);
    auto y4 = conv_mp->forward(x);
    y4 = conv6->forward(y4);
    auto z = _Concat({y1, y2, y3, y4}, 1);
    return {z};
}

_fireMoudleNoBN::_fireMoudleNoBN(int inputChannel, int squeeze_1x1, int expand_1x1, int expand_3x3) {
    NN::ConvOption convOption;
    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannel, squeeze_1x1};
    convOption.padMode    = Express::VALID;
    conv1.reset(NN::Conv(convOption));

    convOption.kernelSize = {1, 1};
    convOption.channel    = {squeeze_1x1, expand_1x1};
    convOption.padMode    = Express::VALID;
    conv2.reset(NN::Conv(convOption));

    convOption.kernelSize = {3, 3};
    convOption.channel    = {squeeze_1x1, expand_3x3};
    convOption.padMode    = Express::SAME;
    conv3.reset(NN::Conv(convOption));

    registerModel({conv1, conv2, conv3});
}

std::vector<Express::VARP> _fireMoudleNoBN::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x = conv1->forward(x);
    auto y1 = conv2->forward(x);
    auto y2 = conv3->forward(x);
    auto z  = _Concat({y1, y2}, 1);
    return {z};
}

_IdentityBlockNoBN::_IdentityBlockNoBN(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels) {
    cbr1 = ConvBnReluNoBN({inputChannels[0], outputChannels[0]}, 1, 1, false);
    cbr2 = ConvBnReluNoBN({inputChannels[1], outputChannels[1]}, kernelSize, 1, false);

    NN::ConvOption convOption;
    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannels[2], outputChannels[2]};
    convOption.stride     = {1, 1};
    convOption.padMode    = SAME;
    conv.reset(NN::Conv(convOption));
    registerModel({cbr1, cbr2, conv});
}

std::vector<Express::VARP> _IdentityBlockNoBN::onForward(const std::vector<Express::VARP> &inputs) {
    VARP input_tensor = inputs[0];
    auto x = cbr1->forward(input_tensor);
    x      = cbr2->forward(x);
    x      = conv->forward(x);
    x      = _Add(x, input_tensor);
    x      = _Relu(x);
    return {x};
}

_Resnet50ConvBlockNoBN::_Resnet50ConvBlockNoBN(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels, int stride) {
    cbr1 = ConvBnReluNoBN({inputChannels[0], outputChannels[0]}, 1, stride);
    cbr2 = ConvBnReluNoBN({inputChannels[1], outputChannels[1]}, kernelSize, 1);

    NN::ConvOption convOption;
    convOption.kernelSize = {1, 1};
    convOption.padMode    = SAME;

    convOption.channel    = {inputChannels[2], outputChannels[2]};
    convOption.stride     = {1, 1};
    conv.reset(NN::Conv(convOption));

    convOption.channel    = {inputChannels[0], outputChannels[2]};
    convOption.stride     = {stride, stride};
    conv_sc.reset(NN::Conv(convOption));

    registerModel({cbr1, cbr2, conv, conv_sc});
}

std::vector<Express::VARP> _Resnet50ConvBlockNoBN::onForward(const std::vector<Express::VARP> &inputs) {
    VARP input_tensor = inputs[0];
    auto x = cbr1->forward(input_tensor);
    x = cbr2->forward(x);
    x = conv->forward(x);

    auto shortcut = conv_sc->forward(input_tensor);
    x = _Add(x, shortcut);
    x = _Relu(x);
    return {x};
}

_Resnet18BasicBlockNoBN::_Resnet18BasicBlockNoBN(int inputChannel, int outputChannel, int kernelSize, int stride, bool convShortcut) {
    NN::ConvOption convOption;
    convOption.kernelSize = {kernelSize, kernelSize};
    convOption.padMode    = SAME;

    convOption.channel    = {inputChannel, outputChannel};
    convOption.stride     = {stride, stride};

    conv1.reset(NN::Conv(convOption));

    convOption.channel    = {outputChannel, outputChannel};
    convOption.stride     = {1, 1};
    conv2.reset(NN::Conv(convOption));

    useConvShortcut = convShortcut;

    if (convShortcut) {
        convOption.channel    = {inputChannel, outputChannel};
        convOption.stride     = {stride, stride};
        conv_sc.reset(NN::Conv(convOption));
        registerModel({conv1, conv2, conv_sc});
    } else {
        registerModel({conv1, conv2});
    }
}

std::vector<Express::VARP> _Resnet18BasicBlockNoBN::onForward(const std::vector<Express::VARP> &inputs) {
    VARP input_tensor = inputs[0];
    auto x = conv1->forward(input_tensor);
    x = conv2->forward(x);
    if (useConvShortcut) {
        auto sc = conv_sc->forward(input_tensor);
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