//
// Created by fss on 23-6-25.
//
#include "runtime/ir.h"
#include "runtime/runtime_ir.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>

TEST(test_ir, topo)
{
    using namespace jinfer;
    std::string bin_path("model_file/resnet18_batch1.pnnx.bin");
    std::string param_path("model_file/resnet18_batch1.param");
    RuntimeGraph graph(param_path, bin_path);
    const bool init_success = graph.init();
    ASSERT_EQ(init_success,
              true);
    graph.build("pnnx_input_0", "pnnx_output_0");
    const auto &topo_queues = graph.get_topo_seq();

    int index = 0;
    for (
        const auto &operator_ : topo_queues) {
        LOG(INFO)
            << "Index: " << index << " Type: " << operator_->type
            << " Name: " << operator_->name;
        index += 1;
    }
}

TEST(test_ir, build_output_ops)
{
    using namespace jinfer;
    std::string bin_path("model_file/simple_ops.pnnx.bin");
    std::string param_path("model_file/simple_ops.pnnx.param");
    RuntimeGraph graph(param_path, bin_path);
    const bool init_success = graph.init();
    ASSERT_EQ(init_success,
              true);
    graph.build("pnnx_input_0", "pnnx_output_0");
    const auto &topo_queues = graph.get_topo_seq();

    int index = 0;
    for (
        const auto &operator_ : topo_queues) {
        LOG(INFO)
            << "Index: " << index << " Name: " << operator_->name;
        index += 1;
    }
}

TEST(test_ir, build_output_ops2)
{
    using namespace jinfer;
    std::string bin_path("model_file/simple_ops.pnnx.bin");
    std::string param_path("model_file/simple_ops.pnnx.param");
    RuntimeGraph graph(param_path, bin_path);
    const bool init_success = graph.init();
    ASSERT_EQ(init_success,
              true);
    graph.build("pnnx_input_0", "pnnx_output_0");
    const auto &topo_queues = graph.get_topo_seq();

    int index = 0;
    for (
        const auto &operator_ : topo_queues) {
        LOG(INFO)
            << "operator name: " << operator_->name;
        for (
            const auto &pair : operator_->output_operators) {
            LOG(INFO)
                << "output: " << pair.first;
        }
        LOG(INFO)
            << "-------------------------";
        index += 1;
    }
}

TEST(test_ir, build1_status)
{
    using namespace jinfer;
    std::string bin_path("model_file/simple_ops.pnnx.bin");
    std::string param_path("model_file/simple_ops.pnnx.param");
    RuntimeGraph graph(param_path, bin_path);
    ASSERT_EQ(int(graph.state()), -2);
    const bool init_success = graph.init();
    ASSERT_EQ(init_success,
              true);
    ASSERT_EQ(int(graph.state()), -1);
    graph.build("pnnx_input_0", "pnnx_output_0");
    ASSERT_EQ(int(graph.state()), 0);
}

TEST(test_ir, build1_output_tensors)
{
    using namespace jinfer;
    std::string bin_path("model_file/simple_ops2.pnnx.bin");
    std::string param_path("model_file/simple_ops2.pnnx.param");
    RuntimeGraph graph(param_path, bin_path);
    ASSERT_EQ(int(graph.state()), -2);
    const bool init_success = graph.init();
    ASSERT_EQ(init_success,
              true);
    ASSERT_EQ(int(graph.state()), -1);
    graph.build("pnnx_input_0", "pnnx_output_0");
    ASSERT_EQ(int(graph.state()), 0);

    const auto &ops = graph.operators();
    for (
        const auto &op : ops) {
        LOG(INFO)
            << op->name;
        // 打印op输出空间的张量
        const auto &operand = op->output_operand;
        if (!operand || operand->data.empty()) {
            continue;
        }
        const uint32_t batch_size = operand->data.size();
        LOG(INFO)
            << "batch: " << batch_size;

        for (
            uint32_t i = 0;
            i < batch_size;
            ++i) {
            const auto &data = operand->data.at(i);
            LOG(INFO)
                << "channel: " << data->channels()
                << " height: " << data->rows()
                << " cols: " << data->cols();
        }
    }
}
