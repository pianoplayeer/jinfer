//
// Created by 27836 on 2025/6/15.
//

#include <runtime/runtime_ir.hpp>

namespace jinfer
{

RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
	: param_path_(std::move(param_path)), bin_path_(std::move(bin_path))
{

}

void
RuntimeGraph::set_bin_path(const std::string &bin_path)
{
	this->bin_path_ = bin_path;
}

void
RuntimeGraph::set_param_path(const std::string &param_path)
{
	this->param_path_ = param_path;
}

const std::string &
RuntimeGraph::param_path() const
{
	return this->param_path_;
}

const std::string &
RuntimeGraph::bin_path() const
{
	return this->bin_path_;
}

bool
RuntimeGraph::init()
{
	if (this->bin_path_.empty() || this->param_path_.empty()) {
		LOG(ERROR) << "The bin path or param path is empty";
		return false;
	}

	this->graph_ = std::make_unique<pnnx::Graph>();
	int load_result = this->graph_->load(this->param_path_, this->bin_path_);
	if (load_result != 0) {
		LOG(ERROR) << "Cannot find the param path or bin path: " << this->param_path_
				   << " " << this->bin_path_;
		return false;
	}

	std::vector<pnnx::Operator*> operators = this->graph_->ops;
	if (operators.empty()) {
		LOG(ERROR) << "Cannot read the layers' definition";
		return false;
	}

	this->operators_.clear();
	this->operators_map_.clear();

	for (auto* op: operators) {
		if (!op) {
			LOG(WARNING) << "Meet empty operator";
			continue;
		}

		std::shared_ptr<RuntimeOperator> runtime_operator =
			std::make_shared<RuntimeOperator>();
		runtime_operator->name = op->name;
		runtime_operator->type = op->type;

		if (!op->inputs.empty()) {
			InitGraphOperatorsInput(op->inputs, runtime_operator);
		}

		if (!op->outputs.empty()) {
			InitGraphOperatorsOutput(op->outputs, runtime_operator);
		}

		if (!op->attrs.empty()) {
			InitGraphAttrs(op->attrs, runtime_operator);
		}

		if (!op->params.empty()) {
			InitGraphParams(op->params, runtime_operator);
		}

		this->operators_.push_back(runtime_operator);
		this->operators_map_.insert({ op->name, runtime_operator });
	}

	return true;
}

const std::vector<std::shared_ptr<RuntimeOperator>> &
RuntimeGraph::operators() const
{
	return this->operators_;
}

void
RuntimeGraph::InitGraphOperatorsInput(const std::vector<pnnx::Operand*> &inputs,
	const std::shared_ptr<RuntimeOperator> &runtime_operator)
{
	for (auto* input: inputs) {
		std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();
		runtime_operand->name = input->producer->name;
		runtime_operand->shape = input->shape;

		switch (input->type) {
		case 1: {
			runtime_operand->type = RuntimeDataType::kTypeFloat32;
			break;
		}

		case 0: {
			runtime_operand->type = RuntimeDataType::kTypeUnknown;
			break;
		}

		default: {
			LOG(FATAL) << "Unsupported input operand type: " << input->type;
			break;
		}
		}

		runtime_operator->input_operands_seq.push_back(runtime_operand);
		runtime_operator->input_operands.insert({ runtime_operand->name, runtime_operand });
	}
}

void
RuntimeGraph::InitGraphOperatorsOutput(const std::vector<pnnx::Operand*> &outputs,
	const std::shared_ptr<RuntimeOperator> &runtime_operator)
{
	for (auto* output: outputs) {
		if (!output) {
			continue;
		}

		for (const auto* consumer: output->consumers) {
			runtime_operator->output_names.push_back(consumer->name);
		}
	}
}

void
RuntimeGraph::InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
	const std::shared_ptr<RuntimeOperator> &runtime_operator)
{
	for (const auto &[name, attr]: attrs) {
		switch (attr.type) {
			// float32
		case 1: {
			std::shared_ptr<RuntimeAttribute> runtime_attr =
				std::make_shared<RuntimeAttribute>();
			runtime_attr->type = RuntimeDataType::kTypeFloat32;
			runtime_attr->weight_data = attr.data;
			runtime_attr->shape = attr.shape;
			runtime_operator->attrs.insert({ name, runtime_attr });
			break;
		}
		default: {
			LOG(FATAL) << "Init Graph error, unsupported attribute type: " << attr.type;
			break;
		}
		}
	}
}

void
RuntimeGraph::InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,
	const std::shared_ptr<RuntimeOperator> &runtime_operator)
{
	for (const auto &[name, param]: params) {
		if (name.empty()) {
			continue;
		}

		const int type = param.type;
		switch (type) {
		case int(RuntimeParameterType::kParameterUnknown): {
			std::shared_ptr<RuntimeParameter> runtime_parameter = std::make_shared<RuntimeParameter>();
			runtime_operator->params.insert({ name, runtime_parameter });
			break;
		}

		case int(RuntimeParameterType::kParameterBool): {
			std::shared_ptr<RuntimeParameterBool> runtime_parameter = std::make_shared<RuntimeParameterBool>();
			runtime_parameter->value = param.b;
			runtime_operator->params.insert({ name, runtime_parameter });
			break;
		}

		case int(RuntimeParameterType::kParameterInt): {
			std::shared_ptr<RuntimeParameterInt> runtime_parameter = std::make_shared<RuntimeParameterInt>();
			runtime_parameter->value = param.i;
			runtime_operator->params.insert({ name, runtime_parameter });
			break;
		}

		case int(RuntimeParameterType::kParameterFloat): {
			std::shared_ptr<RuntimeParameterFloat> runtime_parameter =
				std::make_shared<RuntimeParameterFloat>();
			runtime_parameter->value = param.f;
			runtime_operator->params.insert({ name, runtime_parameter });
			break;
		}

		case int(RuntimeParameterType::kParameterString): {
			std::shared_ptr<RuntimeParameterString> runtime_parameter =
				std::make_shared<RuntimeParameterString>();
			runtime_parameter->value = param.s;
			runtime_operator->params.insert({ name, runtime_parameter });
			break;
		}

		case int(RuntimeParameterType::kParameterIntArray): {
			std::shared_ptr<RuntimeParameterIntArray> runtime_parameter =
				std::make_shared<RuntimeParameterIntArray>();
			runtime_parameter->value = param.ai;
			runtime_operator->params.insert({ name, runtime_parameter });
			break;
		}

		case int(RuntimeParameterType::kParameterFloatArray): {
			std::shared_ptr<RuntimeParameterFloatArray> runtime_parameter =
				std::make_shared<RuntimeParameterFloatArray>();
			runtime_parameter->value = param.af;
			runtime_operator->params.insert({ name, runtime_parameter });
			break;
		}

		case int(RuntimeParameterType::kParameterStringArray): {
			std::shared_ptr<RuntimeParameterStringArray> runtime_parameter =
				std::make_shared<RuntimeParameterStringArray>();
			runtime_parameter->value = param.as;
			runtime_operator->params.insert({ name, runtime_parameter });
			break;
		}

		default: {
			LOG(FATAL) << "Unknown parameter type: " << type;
			break;
		}

		}
	}
}

}