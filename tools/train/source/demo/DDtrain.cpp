//
//  DDtrain.cpp
//  MNN
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Executor.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include "DemoUnit.hpp"
#include "DDtrainUtils.hpp"
#include "NN.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "module/PipelineModule.hpp"
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "ConvNet.hpp"
#include "MobilenetV2NoBN.hpp"

using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

class DDtrain : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 3) {
            std::cout << "usage: ./runTrainDemo.out MobilenetV2Train path/to/train/images/ path/to/train/image/txt path/to/test/images/ path/to/test/image/txt" << std::endl;
            return 0;
        }
        Executor::getGlobalExecutor()->setLazyComputeMode(MNN::Express::Executor::LAZY_FULL);
        // global random number generator, should invoke before construct the model and dataset
        RandomGenerator::generator(17);

        std::string syn_root = argv[1];
        std::string syn_pathToImageTxt = argv[2];
        std::string tset_root = argv[3];
        // First, No BN
        
        std::string model_name = "convnet";
        
        DDtrainUtils::train(model_name,syn_root, syn_pathToImageTxt, tset_root);

        return 0;
    }
};
DemoUnitSetRegister(DDtrain, "DDtrain");

