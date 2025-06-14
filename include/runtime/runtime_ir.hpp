//
// Created by 27836 on 2025/6/15.
//

#ifndef _RUNTIME_IR_HPP_
#define _RUNTIME_IR_HPP_

#include <string>
#include <vector>
#include <map>
#include "ir.h"
#include "runtime_operator.hpp"

namespace jinfer
{

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
	void set_bin_path(const std::string &bin_path);

	/**
	 * 设置结构文件
	 * @param param_path  结构文件路径
	 */
	void set_param_path(const std::string &param_path);

	/**
	 * 返回结构文件
	 * @return 返回结构文件
	 */
	const std::string &param_path() const;

	/**
	 * 返回权重文件
	 * @return 返回权重文件
	 */
	const std::string &bin_path() const;

	/**
	 * 计算图的初始化
	 * @return 是否初始化成功
	 */
	bool init();

	const std::vector<std::shared_ptr<RuntimeOperator>> &operators() const;

private:
	/**
	 * 初始化kuiper infer计算图节点中的输入操作数
	 * @param inputs pnnx中的输入操作数
	 * @param runtime_operator 计算图节点
	 */
	static void InitGraphOperatorsInput(
		const std::vector<pnnx::Operand*> &inputs,
		const std::shared_ptr<RuntimeOperator> &runtime_operator);

	/**
	 * 初始化kuiper infer计算图节点中的输出操作数
	 * @param outputs pnnx中的输出操作数
	 * @param runtime_operator 计算图节点
	 */
	static void InitGraphOperatorsOutput(
		const std::vector<pnnx::Operand*> &outputs,
		const std::shared_ptr<RuntimeOperator> &runtime_operator);

	/**
	 * 初始化kuiper infer计算图中的节点属性
	 * @param attrs pnnx中的节点属性
	 * @param runtime_operator 计算图节点
	 */
	static void
	InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
		const std::shared_ptr<RuntimeOperator> &runtime_operator);

	/**
	 * 初始化kuiper infer计算图中的节点参数
	 * @param params pnnx中的参数属性
	 * @param runtime_operator 计算图节点
	 */
	static void
	InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,
		const std::shared_ptr<RuntimeOperator> &runtime_operator);

private:
	std::string input_name_;
	std::string output_name_;
	std::string bin_path_;
	std::string param_path_;

	std::vector<std::shared_ptr<RuntimeOperator>> operators_;
	std::map<std::string, std::shared_ptr<RuntimeOperator>> operators_map_;
	std::unique_ptr<pnnx::Graph> graph_;

};

}

#endif //_RUNTIME_IR_HPP_
