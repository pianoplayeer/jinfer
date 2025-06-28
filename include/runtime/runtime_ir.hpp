//
// Created by 27836 on 2025/6/15.
//

#ifndef _RUNTIME_IR_HPP_
#define _RUNTIME_IR_HPP_

#include "ir.h"
#include "runtime_operator.hpp"
#include <map>
#include <string>
#include <vector>

namespace jinfer
{

// need_init -> need_build -> completed
enum class GraphState {
    need_init = -2,
    need_build = -1,
    completed = 0
};

class RuntimeGraph
{

public:
    /**
   * 初始化计算图
   * @param param_path 计算图的结构文件
   * @param bin_path 计算图中的权重文件
   */
    RuntimeGraph(std::string param_path, std::string bin_path);

    /**
     * 设置权重文件
     * @param bin_path 权重文件路径
     */
    void
    set_bin_path(const std::string &bin_path);

    /**
     * 设置结构文件
     * @param param_path  结构文件路径
     */
    void
    set_param_path(const std::string &param_path);

    /**
     * 返回结构文件
     * @return 返回结构文件
     */
    const std::string &
    param_path() const;

    /**
     * 返回权重文件
     * @return 返回权重文件
     */
    const std::string &
    bin_path() const;

    /**
     * 计算图的初始化
     * @return 是否初始化成功
     */
    bool
    init();

    bool
    build(std::string input_op_name, std::string output_op_name);

    const std::vector<std::shared_ptr<RuntimeOperator>> &
    operators() const;

    GraphState
    state() const;

    const std::vector<std::shared_ptr<RuntimeOperator>> &get_topo_seq() const;

private:
    /**
     * 初始化kuiper infer计算图节点中的输入操作数
     * @param inputs pnnx中的输入操作数
     * @param runtime_operator 计算图节点
     */
    void
    init_input_ops(
        const std::vector<pnnx::Operand *> &inputs,
        const std::shared_ptr<RuntimeOperator> &runtime_operator);

    /**
     * 初始化kuiper infer计算图节点中的输出操作数
     * @param outputs pnnx中的输出操作数
     * @param runtime_operator 计算图节点
     */
    void
    init_output_ops(
        const std::vector<pnnx::Operand *> &outputs,
        const std::shared_ptr<RuntimeOperator> &runtime_operator);

    /**
     * 初始化kuiper infer计算图中的节点属性
     * @param attrs pnnx中的节点属性
     * @param runtime_operator 计算图节点
     */
    void
    init_op_attrs(const std::map<std::string, pnnx::Attribute> &attrs,
                  const std::shared_ptr<RuntimeOperator> &runtime_operator);

    /**
     * 初始化kuiper infer计算图中的节点参数
     * @param params pnnx中的参数属性
     * @param runtime_operator 计算图节点
     */
    void
    init_op_params(const std::map<std::string, pnnx::Parameter> &params,
                   const std::shared_ptr<RuntimeOperator> &runtime_operator);

    std::shared_ptr<RuntimeOperator>
    create_op(const std::string& name);

    void
    init_topo_seq(const std::shared_ptr<RuntimeOperator> &start);

    void
    reverse_topo(const std::shared_ptr<RuntimeOperator> &cur);

    void
    check_shape(const std::vector<int> &shape) const;

    void
    init_data(std::vector<std::shared_ptr<Tensor<float>>> &data,
              const std::vector<int> &shape);

private:
    std::string input_name_;
    std::string output_name_;
    std::string bin_path_;
    std::string param_path_;

    GraphState graph_state_ = GraphState::need_init;
    std::vector<std::shared_ptr<RuntimeOperator>> operators_;
    std::vector<std::shared_ptr<RuntimeOperator>> topo_operators_;
    std::map<std::string, std::shared_ptr<RuntimeOperator>> operators_map_;
    std::unique_ptr<pnnx::Graph> graph_;
};

}// namespace jinfer

#endif//_RUNTIME_IR_HPP_
