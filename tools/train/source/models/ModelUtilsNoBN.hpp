//
//  ModelUtilsNoBN.hpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ModelUtilsNoBN_hpp
#define ModelUtilsNoBN_hpp

#include <MNN/expr/Module.hpp>
#include "NN.hpp"

namespace MNN {
namespace Train {
namespace Model {
using namespace Express;
// https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

class _DepthwiseSeparableConv2DNoBN : public Module {
public:
    _DepthwiseSeparableConv2DNoBN(std::vector<int> inputOutputChannels, int stride);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv3x3;
    std::shared_ptr<Module> conv1x1;
};

class _BottleNeckNoBN : public Module {
public:
    _BottleNeckNoBN(std::vector<int> inputOutputChannels, int stride, int expandRatio);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::vector<std::shared_ptr<Module> > layers;
    bool useShortcut = false;
};

class _ConvBnReluNoBN : public Module {
public:
    _ConvBnReluNoBN(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, bool depthwise = false, PaddingMode padmod = SAME);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv;
};

class _InceptionNoBN : public Module {
public:
    _InceptionNoBN(int inputChannelSet, int channel_1x1,
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
    std::shared_ptr<Module> conv_mp;
};

class _fireMoudleNoBN : public Module {
public:
    _fireMoudleNoBN(int inputChannel, int squeeze_1x1, int expand_1x1, int expand_3x3);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> conv2;
    std::shared_ptr<Module> conv3;

};

class _IdentityBlockNoBN : public Module {
public:
    _IdentityBlockNoBN(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module>cbr1, cbr2;
    std::shared_ptr<Module>conv;

};

class _Resnet50ConvBlockNoBN: public Module {
public:
    _Resnet50ConvBlockNoBN(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels, int stride=2);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
    std::shared_ptr<Module>cbr1, cbr2;
    std::shared_ptr<Module>conv, conv_sc;
};

class _Resnet18BasicBlockNoBN: public Module {
public:
    _Resnet18BasicBlockNoBN(int inputChannel, int outputChannel, int kernelSize, int stride=1, bool convShortcut=false);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
    std::shared_ptr<Module>conv1, conv2;
    std::shared_ptr<Module>conv_sc;
    bool useConvShortcut;
};

std::shared_ptr<Module> DepthwiseSeparableConv2DNoBN(std::vector<int> inputOutputChannels, int stride= 1);
std::shared_ptr<Module> ConvBnReluNoBN(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, bool depthwise = false, PaddingMode padmod = SAME);
std::shared_ptr<Module> BottleNeckNoBN(std::vector<int> inputOutputChannels, int stride, int expandRatio);
std::shared_ptr<Module> InceptionNoBN(int inputChannelSet, int channel_1x1, int channel_3x3_reduce, int channel_3x3,
                                  int channel_5x5_reduce, int channel_5x5, int channel_pool);
std::shared_ptr<Module> FireMoudleNoBN(int inputChannel, int squeeze_1x1, int expand_1x1, int expand_3x3);
std::shared_ptr<Module> IdentityBlockNoBN(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels);
std::shared_ptr<Module> Resnet50ConvBlockNoBN(int kernelSize, std::vector<int> inputChannels, std::vector<int> outputChannels, int stride= 2);
std::shared_ptr<Module> Resnet18BasicBlockNoBN(int inputChannel, int outputChannel, int kernelSize=3, int stride=1, bool convShortcut=false);

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // ModelUtilsNoBN_hpp
