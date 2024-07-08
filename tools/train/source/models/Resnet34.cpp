//
//  .cpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef Resnet34_cpp
#define Resnet34_cpp

#include "Resnet34.hpp"
#include "Initializer.hpp"
#include "NN.hpp"
using namespace MNN::Express;

namespace MNN {
namespace Train {
namespace Model {

Resnet34::Resnet34(int numClass) {
    NN::ConvOption convOption;
    convOption.padMode = SAME;
    convOption.kernelSize = {7, 7};
    convOption.stride = {2, 2};
    convOption.channel = {3, 64};
    conv1.reset(NN::Conv(convOption));
    bn1.reset(NN::BatchNorm(64));

    convOption.channel = {64, 64};
    convOption.stride = {2, 2};
    convOption.kernelSize = {3, 3};
    convOption.depthwise  =true;
    conv_mp.reset(NN::Conv(convOption));

    convOption.channel = {512, 512};
    convOption.stride = {7, 7};
    convOption.kernelSize = {7, 7};
    conv_ap.reset(NN::Conv(convOption));

    layer11 = Resnet18BasicBlock(64, 64);
    layer12 = Resnet18BasicBlock(64, 64);
    layer13 = Resnet18BasicBlock(64, 64);

    layer21 = Resnet18BasicBlock(64, 128, 3, 2, true);
    layer22 = Resnet18BasicBlock(128, 128);
    layer23 = Resnet18BasicBlock(128, 128);
    layer24 = Resnet18BasicBlock(128, 128);

    layer31 = Resnet18BasicBlock(128, 256, 3, 2, true);
    layer32 = Resnet18BasicBlock(256, 256);
    layer33 = Resnet18BasicBlock(256, 256);
    layer34 = Resnet18BasicBlock(256, 256);
    layer35 = Resnet18BasicBlock(256, 256);
    layer36 = Resnet18BasicBlock(256, 256);

    layer41 = Resnet18BasicBlock(256, 512, 3, 2, true);
    layer42 = Resnet18BasicBlock(512, 512);
    layer43 = Resnet18BasicBlock(512, 512);

    fc.reset(NN::Linear(512, numClass, true));

    registerModel({conv1, bn1, conv_mp, conv_ap, fc,
                   layer11, layer12, layer13,
                   layer21, layer22, layer23, layer24,
                   layer31, layer32, layer33, layer34, layer35, layer36,
                   layer41, layer42, layer43,
                  });
    setName("Resnet34");
}

std::vector<Express::VARP> Resnet34::onForward(const std::vector<Express::VARP>& inputs) {
    auto x = inputs[0];
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = conv_mp->forward(x);

    x = layer11->forward(x);
    x = layer12->forward(x);
    x = layer13->forward(x);

    x = layer21->forward(x);
    x = layer22->forward(x);
    x = layer23->forward(x);
    x = layer24->forward(x);

    x = layer31->forward(x);
    x = layer32->forward(x);
    x = layer33->forward(x);
    x = layer34->forward(x);
    x = layer35->forward(x);
    x = layer36->forward(x);

    x = layer41->forward(x);
    x = layer42->forward(x);
    x = layer43->forward(x);

    x = conv_ap->forward(x);
    x = _Convert(x, NCHW);
    x = _Reshape(x, {0, -1});
    x = fc->forward(x);
    x = _Softmax(x, 1);
    return {x};
}


} // namespace Model
} // namespace Train
} // namespace MNN
#endif