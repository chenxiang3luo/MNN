//
//  mobilenetV2Train.cpp
//  MNN
//
//  Created by MNN on 2020/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include "DemoUnit.hpp"
#include "MobilenetV2NoBN.hpp"
#include "DataDistillationUtils.hpp"
#include "NN.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "module/PipelineModule.hpp"

using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

class DataDistillationTrain : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 2) {
            std::cout << "usage: ./runTrainDemo.out MobilenetV2Train path/to/train/images/ path/to/train/image/txt path/to/test/images/ path/to/test/image/txt" << std::endl;
            return 0;
        }
        // global random number generator, should invoke before construct the model and dataset
        RandomGenerator::generator(17);

        std::string root = argv[1];
        // First, No BN
        std::shared_ptr<MNN::Train::Model::MobilenetV2NoBN> model(new MobilenetV2NoBN(10));
        
        DataDistillationUtils::train(model, 10, 1,root);

        return 0;
    }
};

DemoUnitSetRegister(DataDistillationTrain, "DataDistillationTrain");

