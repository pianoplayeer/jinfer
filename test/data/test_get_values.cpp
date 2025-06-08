//
// Created by fss on 23-6-4.
//

#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
TEST(test_tensor_values, tensor_values1)
{
	using namespace jinfer;
	Tensor<float> f1(2, 3, 4);
	f1.rand();
	f1.show();

	LOG(INFO) << "Data in the first channel: " << f1.slice(0);
	LOG(INFO) << "Data in the (1,1,1): " << f1.at(1, 1, 1);
}