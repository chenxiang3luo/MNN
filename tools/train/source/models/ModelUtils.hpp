//
//  ModelUtils.hpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ModelUtils_hpp
#define ModelUtils_hpp

#include <MNN/expr/Module.hpp>
#include "NN.hpp"

namespace MNN {
namespace Train {
namespace Model {
using namespace Express;
// https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
int makeDivisible(int v, int divisor = 8, int minValue = 0);

class _DepthwiseSeparableConv2D : public Module {
public:
    _DepthwiseSeparableConv2D(std::vector<int> inputOutputChannels, int stride);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv3x3;
    std::shared_ptr<Module> bn1;
    std::shared_ptr<Module> conv1x1;
    std::shared_ptr<Module> bn2;
};

class _BottleNeck : public Module {
public:
    _BottleNeck(std::vector<int> inputOutputChannels, int stride, int expandRatio);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::vector<std::shared_ptr<Module> > layers;
    bool useShortcut = false;
};

class _ConvBnRelu : public Module {
public:
    _ConvBnRelu(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, bool depthwise = false, PaddingMode padmod = SAME);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv;
    std::shared_ptr<Module> bn;
};

class _Inception : public Module {
public:
    _Inception(int inputChannelSet, int channel_1x1,
               int channel_3x3_reduce, int channel_3x3,
               int channel_5x5_reduce, int channel_5x5,
               int channel_pool);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> conv2;
    std::shared_ptr<Module> conv3;
    std::shared_ptr<Module> conv4;
    std::shared_ptr<Module> conv5;
    std::shared_ptr<Module> conv6;
    std::shared_ptr<Module> bn1, bn2, bn3, bn4, bn5, bn6;
    std::shared_ptr<Module> conv_mp;
};

class _fireMoudle : public Module {
public:
    _fireMoudle(int inputChannel, int squeeze_1x1, int expand_1x1, int expand_3x3);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> conv2;
    std::shared_ptr<Module> conv3;
    std::shared_ptr<Module> bn1, bn2, bn3;

};

class _IdentityBlock : public Module {
public:
    _IdentityBlock(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module>cbr1, cbr2;
    std::shared_ptr<Module>conv, bn;

};

class _Resnet50ConvBlock: public Module {
public:
    _Resnet50ConvBlock(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels, int stride=2);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
    std::shared_ptr<Module>cbr1, cbr2;
    std::shared_ptr<Module>conv, bn, conv_sc, bn_sc;
};

class _Resnet18BasicBlock: public Module {
public:
    _Resnet18BasicBlock(int inputChannel, int outputChannel, int kernelSize, int stride=1, bool convShortcut=false);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
    std::shared_ptr<Module>conv1, conv2, bn1, bn2;
    std::shared_ptr<Module>conv_sc, bn_sc;
    bool useConvShortcut;
};

std::shared_ptr<Module> DepthwiseSeparableConv2D(std::vector<int> inputOutputChannels, int stride= 1);
std::shared_ptr<Module> ConvBnRelu(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, bool depthwise = false, PaddingMode padmod = SAME);
std::shared_ptr<Module> BottleNeck(std::vector<int> inputOutputChannels, int stride, int expandRatio);
std::shared_ptr<Module> Inception(int inputChannelSet, int channel_1x1, int channel_3x3_reduce, int channel_3x3,
                                  int channel_5x5_reduce, int channel_5x5, int channel_pool);
std::shared_ptr<Module> FireMoudle(int inputChannel, int squeeze_1x1, int expand_1x1, int expand_3x3);
std::shared_ptr<Module> IdentityBlock(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels);
std::shared_ptr<Module> Resnet50ConvBlock(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels, int stride= 2);
std::shared_ptr<Module> Resnet18BasicBlock(int inputChannel, int outputChannel, int kernelSize=3, int stride=1, bool convShortcut=false);

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // ModelUtils_hpp
