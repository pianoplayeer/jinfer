//
// Created by 27836 on 2025/6/15.
//

#include <vector>
#include <runtime/runtime_attr.hpp>

namespace jinfer {

template<class T> std::vector<T>
RuntimeAttribute::get(bool need_clear_weight)
{
	CHECK(!this->weight_data.empty());
	CHECK(this->type != RuntimeDataType::kTypeUnknown);

	std::vector<T> weights;
	switch (this->type) {
	case RuntimeDataType::kTypeFloat32: {
		bool same_type = std::is_same<T, float>::value;
		CHECK_EQ(same_type, true);
		const uint32_t float_size = sizeof(float);
		CHECK_EQ(this->weight_data.size() % float_size, 0);

		for (uint32_t i = 0; i < this->weight_data.size() / float_size; i++) {
			float weight = *((float*)weight_data.data() + i);
			weights.push_back(weight);
		}
		break;
	}

	default: {
		LOG(FATAL) << "Unknown weight data type: " << int(type);
	}
	}

	if (need_clear_weight) {
		this->clear_weight();
	}
}

void
RuntimeAttribute::clear_weight()
{
	if (!this->weight_data.empty()) {
		std::vector<char> tmp = std::vector<char>();
		this->weight_data.swap(tmp);
	}
}

}