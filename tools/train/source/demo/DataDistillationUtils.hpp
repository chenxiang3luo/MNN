//
//  MobilenetV2Utils.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DataDistillationUtilsUtils_hpp
#define DataDistillationUtilsUtils_hpp

#include <MNN/expr/Module.hpp>
#include <string>
#include "MobilenetV2.hpp"
#include "MobilenetV2Utils.hpp"
#include "MobilenetV2NoBN.hpp"

class DataDistillationUtils {
public:
    static void train(std::string model_name, const int numClasses, const int addToLabel,
                      std::string root, const int quantBits = 8);
    
};

#endif
