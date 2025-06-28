//
// Created by 27836 on 2025/6/15.
//

#ifndef _RUNTIME_PARAM_HPP_
#define _RUNTIME_PARAM_HPP_

#include "runtime_datatype.hpp"

namespace jinfer
{

struct RuntimeParameter {
    virtual ~RuntimeParameter() = default;

    explicit RuntimeParameter(RuntimeParameterType type = RuntimeParameterType::kParameterUnknown)
        : type(type)
    {
    }

    RuntimeParameterType type = RuntimeParameterType::kParameterUnknown;
};

struct RuntimeParameterInt: public RuntimeParameter {
    RuntimeParameterInt() : RuntimeParameter(RuntimeParameterType::kParameterInt)
    {
    }
    int value = 0;
};

struct RuntimeParameterFloat: public RuntimeParameter {
    RuntimeParameterFloat() : RuntimeParameter(RuntimeParameterType::kParameterFloat)
    {
    }
    float value = 0.f;
};

struct RuntimeParameterString: public RuntimeParameter {
    RuntimeParameterString() : RuntimeParameter(RuntimeParameterType::kParameterString)
    {
    }
    std::string value;
};

struct RuntimeParameterIntArray: public RuntimeParameter {
    RuntimeParameterIntArray() : RuntimeParameter(RuntimeParameterType::kParameterIntArray)
    {
    }
    std::vector<int> value;
};

struct RuntimeParameterFloatArray: public RuntimeParameter {
    RuntimeParameterFloatArray() : RuntimeParameter(RuntimeParameterType::kParameterFloatArray)
    {
    }
    std::vector<float> value;
};

struct RuntimeParameterStringArray: public RuntimeParameter {
    RuntimeParameterStringArray() : RuntimeParameter(RuntimeParameterType::kParameterStringArray)
    {
    }
    std::vector<std::string> value;
};

struct RuntimeParameterBool: public RuntimeParameter {
    RuntimeParameterBool() : RuntimeParameter(RuntimeParameterType::kParameterBool)
    {
    }
    bool value = false;
};

}// namespace jinfer

#endif//_RUNTIME_PARAM_HPP_
