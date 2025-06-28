//
// Created by fss on 23-6-4.
//
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

float MinusOne(float value)
{
    return value - 1.f;
}
TEST(test_transform, transform1)
{
    using namespace jinfer;
    Tensor<float> f1(2, 3, 4);
    f1.rand();
    f1.show();
    f1.transform(MinusOne);
    f1.show();
}