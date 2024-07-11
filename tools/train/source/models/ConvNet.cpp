//
//  .cpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef ConvNet_cpp
#define ConvNet_cpp

#include "ConvNet.hpp"
#include "Initializer.hpp"
#include "NN.hpp"
#include <iostream>

using namespace MNN::Express;



namespace MNN {
namespace Train {
namespace Model {
ConvNet::ConvNet(int numClass, int inputChannels,int net_width, int net_depth, int im_size) {
    for (int i = 0; i < net_depth;i++){
        
        convBlocks.emplace_back(ConvBnRelu({inputChannels, net_width}, 3, 1));
        inputChannels = net_width;
    }

    fc.reset(NN::Linear(27, numClass, true, std::shared_ptr<Initializer>(Initializer::MSRA())));
    registerModel(convBlocks);
    registerModel({fc});
    setName("ConvNet");
}

std::vector<Express::VARP> ConvNet::onForward(const std::vector<Express::VARP>& inputs) {
    auto x = inputs[0];
    auto net_depth = convBlocks.size();
    for (int i = 0; i < net_depth;i++){
        x = convBlocks[i]->forward(x);
        x = _AvePool(x,{2,2},{2,2});
    }
    
    x = _Convert(x, NCHW);
    x = _Reshape(x, {0, -1});
    
    x = fc->forward(x);
    x = _Softmax(x, 1);
    // std::cout<<x->getInfo()->order<<std::endl;
    return {x};
}


std::vector<Express::VARP> ConvNet::onEmbedding(const std::vector<Express::VARP>& inputs) {
    auto x = inputs[0];
    auto net_depth = convBlocks.size();
    for (int i = 0; i < net_depth;i++){
        x = convBlocks[i]->forward(x);
        x = _AvePool(x,{2,2},{2,2});
    }
    x = _Convert(x, NCHW);
    x = _Reshape(x, {0, -1});
    return {x};
}

Express::VARP ConvNet::embedding(Express::VARP input) {
    return this->onEmbedding({input})[0];
}

} // namespace Model
} // namespace Train
} // namespace MNN
#endif