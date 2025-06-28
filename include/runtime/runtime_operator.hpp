//
// Created by 27836 on 2025/6/15.
//

#ifndef _RUNTIME_OP_HPP_
#define _RUNTIME_OP_HPP_

#include "runtime_attr.hpp"
#include "runtime_operand.hpp"
#include "runtime_param.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace jinfer
{

class Layer;

struct RuntimeOperator {

    bool has_forward = false;
    std::string name;
    std::string type;
    std::shared_ptr<Layer> layer;

    std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands;
    std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq;

    std::shared_ptr<RuntimeOperand> output_operand;
    std::vector<std::string> output_names;/// 输出节点名称
    std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators;

    std::map<std::string, std::shared_ptr<RuntimeParameter>> params;
    std::map<std::string, std::shared_ptr<RuntimeAttribute>> attrs;
};

}// namespace jinfer

#endif//_RUNTIME_OP_HPP_
