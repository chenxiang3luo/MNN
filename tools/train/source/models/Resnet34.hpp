//
//  Resnet34.hpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Resnet34_hpp
#define Resnet34_hpp

#include <MNN/expr/Module.hpp>
#include "ModelUtils.hpp"

namespace MNN {
namespace Train {
namespace Model {

class MNN_PUBLIC Resnet34 : public Express::Module {
public:
    Resnet34(int numClass=1001);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    std::shared_ptr<Module> conv1, bn1, conv_mp;
    std::shared_ptr<Module> layer11, layer12, layer13;
    std::shared_ptr<Module> layer21, layer22, layer23, layer24;
    std::shared_ptr<Module> layer31, layer32, layer33, layer34, layer35, layer36;
    std::shared_ptr<Module> layer41, layer42, layer43;
    std::shared_ptr<Module> conv_ap, fc;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // Resnet34Models_hpp