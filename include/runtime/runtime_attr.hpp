//
// Created by 27836 on 2025/6/15.
//

#ifndef _RUNTIME_ATTR_HPP_
#define _RUNTIME_ATTR_HPP_

#include <vector>
#include <memory>
#include <glog/logging.h>
#include "runtime_datatype.hpp"

namespace jinfer
{

struct RuntimeAttribute
{
	std::vector<int> shape;
	std::vector<char> weight_data;
	RuntimeDataType type = RuntimeDataType::kTypeUnknown;

	template<class T> std::vector<T>
	get(bool need_clear_weight = true);

	void
	clear_weight();

};

}

#endif //_RUNTIME_ATTR_HPP_
