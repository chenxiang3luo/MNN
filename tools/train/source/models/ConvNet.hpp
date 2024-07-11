//
//  ConvNet.hpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvNet_hpp
#define ConvNet_hpp

#include <MNN/expr/Module.hpp>
#include "ModelUtils.hpp"

namespace MNN {
namespace Train {
namespace Model {

class MNN_PUBLIC ConvNet : public Express::Module {
public:
    ConvNet(int numClass=1001,int inputChannels=1 ,int net_width=128, int net_depth=3, int im_size=28);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    std::vector<Express::VARP> onEmbedding(const std::vector<Express::VARP>& inputs);
    Express::VARP embedding(Express::VARP input);
    std::vector<std::shared_ptr<Express::Module>> convBlocks;
    std::shared_ptr<Express::Module> fc;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // ConvNetModels_hpp