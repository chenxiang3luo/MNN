//
//  DDtrainUtils.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DDtrainUtils_hpp
#define DDtrainUtils_hpp
#include <MNN/expr/Module.hpp>
class DDtrainUtils {
public:
    static void train(std::string model_name, std::string syn_root, std::string syn_pathToImageTxt, std::string test_root);
};
#endif
