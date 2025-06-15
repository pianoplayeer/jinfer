//
// Created by 27836 on 2025/6/15.
//

#ifndef _RUNTIME_DATATYPE_HPP_
#define _RUNTIME_DATATYPE_HPP_

namespace jinfer {

/// 计算节点属性中的权重类型
enum class RuntimeDataType {
	kTypeUnknown = 0,
	kTypeFloat32 = 1,
	kTypeFloat64 = 2,
	kTypeFloat16 = 3,
	kTypeInt32 = 4,
	kTypeInt64 = 5,
	kTypeInt16 = 6,
	kTypeInt8 = 7,
	kTypeUInt8 = 8,
};

enum class RuntimeParameterType {
	kParameterUnknown = 0,
	kParameterBool = 1,
	kParameterInt = 2,

	kParameterFloat = 3,
	kParameterString = 4,
	kParameterIntArray = 5,
	kParameterFloatArray = 6,
	kParameterStringArray = 7,
};

}

#endif //_RUNTIME_DATATYPE_HPP_
