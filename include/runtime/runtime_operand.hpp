//
// Created by 27836 on 2025/6/15.
//

#ifndef _RUNTIME_OPERAND_HPP_
#define _RUNTIME_OPERAND_HPP_

#include "data/tensor.hpp"
#include "runtime_datatype.hpp"
#include <memory>
#include <string>
#include <vector>

namespace jinfer
{

struct RuntimeOperand {
    std::string name;
    RuntimeDataType type = RuntimeDataType::kTypeUnknown;
    std::vector<int> shape;
    std::vector<std::shared_ptr<Tensor<float>>> data;
};

}// namespace jinfer

#endif//_RUNTIME_OPERAND_HPP_
