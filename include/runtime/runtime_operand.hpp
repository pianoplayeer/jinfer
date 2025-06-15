//
// Created by 27836 on 2025/6/15.
//

#ifndef _RUNTIME_OPERAND_HPP_
#define _RUNTIME_OPERAND_HPP_

#include <string>
#include <vector>
#include <memory>
#include "runtime_datatype.hpp"
#include "data/tensor.hpp"

namespace jinfer {

struct RuntimeOperand {
	std::string name;
	RuntimeDataType type = RuntimeDataType::kTypeUnknown;
	std::vector<int> shape;
	std::vector<std::shared_ptr<Tensor<float>>> data;
};

}

#endif //_RUNTIME_OPERAND_HPP_
