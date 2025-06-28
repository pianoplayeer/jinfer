//
// Created by 27836 on 2025/6/15.
//

#ifndef _RUNTIME_ATTR_HPP_
#define _RUNTIME_ATTR_HPP_

#include "runtime_datatype.hpp"
#include <glog/logging.h>
#include <memory>
#include <vector>

namespace jinfer
{

struct RuntimeAttribute {
    std::vector<int> shape;
    std::vector<char> weight_data;
    RuntimeDataType type = RuntimeDataType::kTypeUnknown;

    template<class T>
    std::vector<T>
    get(bool need_clear_weight = true);

    void
    clear_weight();
};

}// namespace jinfer

#endif//_RUNTIME_ATTR_HPP_
