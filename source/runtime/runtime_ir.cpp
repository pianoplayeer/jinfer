//
// Created by 27836 on 2025/6/15.
//

#include <runtime/runtime_ir.hpp>
#include <queue>

namespace jinfer
{

RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    : param_path_(std::move(param_path)), bin_path_(std::move(bin_path))
{
}

void RuntimeGraph::set_bin_path(const std::string &bin_path)
{
    this->bin_path_ = bin_path;
}

void RuntimeGraph::set_param_path(const std::string &param_path)
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

bool RuntimeGraph::init()
{
    switch (this->graph_state_) {
    case GraphState::need_init:
        LOG(INFO) << "start init";
        break;

    case GraphState::need_build:
    case GraphState::completed:
        LOG(INFO) << "graph has been inited";
        return true;

    default:
        LOG(FATAL) << "unknown graph state: " << int(this->graph_state_);
        return false;
    }

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

    std::vector<pnnx::Operator *> operators = this->graph_->ops;
    if (operators.empty()) {
        LOG(ERROR) << "Cannot read the layers' definition";
        return false;
    }

    this->operators_.clear();
    this->operators_map_.clear();

    for (auto *op : operators) {
        if (!op) {
            LOG(WARNING) << "Meet empty operator";
            continue;
        }

        std::shared_ptr<RuntimeOperator> runtime_operator;
        auto iter = operators_map_.find(op->name);
        if (iter != operators_map_.end()) {
            runtime_operator = iter->second;
        } else {
            runtime_operator = this->create_op(op->name);
        }

        runtime_operator->name = op->name;
        runtime_operator->type = op->type;

        if (!op->inputs.empty()) {
            init_input_ops(op->inputs, runtime_operator);
        }

        if (!op->outputs.empty()) {
            init_output_ops(op->outputs, runtime_operator);
        }

        if (!op->attrs.empty()) {
            init_op_attrs(op->attrs, runtime_operator);
        }

        if (!op->params.empty()) {
            init_op_params(op->params, runtime_operator);
        }
    }

    this->graph_state_ = GraphState::need_build;
    return true;
}

const std::vector<std::shared_ptr<RuntimeOperator>> &
RuntimeGraph::operators() const
{
    return this->operators_;
}

void RuntimeGraph::init_input_ops(const std::vector<pnnx::Operand *> &inputs,
                                  const std::shared_ptr<RuntimeOperator> &runtime_operator)
{
    for (auto *input : inputs) {
        std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();
        runtime_operand->name = input->producer->name;
        runtime_operand->shape = input->shape;
        check_shape(input->shape);

        switch (input->type) {
        case 1: {
            runtime_operand->type = RuntimeDataType::kTypeFloat32;
            init_data(runtime_operand->data, input->shape);
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
        runtime_operator->input_operands.insert({runtime_operand->name, runtime_operand});
    }
}

void RuntimeGraph::init_output_ops(const std::vector<pnnx::Operand *> &outputs,
                                   const std::shared_ptr<RuntimeOperator> &runtime_operator)
{
    CHECK(!outputs.empty()) << "no output operand in operator " << runtime_operator->name;
    CHECK(outputs.size() == 1)
        << "jinfer only supports one output operand for each operator, current op: "
        << runtime_operator->name << ", output num: " << outputs.size();

    /// 采用循环：日后可能改成多输出
    for (auto *output : outputs) {
        if (!output) {
            continue;
        }

        check_shape(output->shape);
        runtime_operator->output_operand = std::make_shared<RuntimeOperand>();
        runtime_operator->output_operand->shape = output->shape;
        runtime_operator->output_operand->name = output->name + "_output";

        switch (output->type) {
        case 1: {
            runtime_operator->output_operand->type = RuntimeDataType::kTypeFloat32;
            init_data(runtime_operator->output_operand->data, output->shape);
            break;
        }

        case 0: {
            runtime_operator->output_operand->type = RuntimeDataType::kTypeUnknown;
            break;
        }

        default: {
            LOG(FATAL) << "Unsupported output operand type: " << output->type;
            break;
        }
        }

        for (const auto *consumer : output->consumers) {
            runtime_operator->output_names.push_back(consumer->name);

            std::shared_ptr<RuntimeOperator> output_operator;
            auto iter = operators_map_.find(consumer->name);
            if (iter != operators_map_.end()) {
                output_operator = iter->second;
            } else {
                output_operator = this->create_op(consumer->name);
            }

            runtime_operator->output_operators.insert({consumer->name, output_operator});
        }
    }
}

void RuntimeGraph::init_op_attrs(const std::map<std::string, pnnx::Attribute> &attrs,
                                 const std::shared_ptr<RuntimeOperator> &runtime_operator)
{
    for (const auto &[name, attr] : attrs) {
        switch (attr.type) {
        // float32
        case 1: {
            std::shared_ptr<RuntimeAttribute> runtime_attr =
                std::make_shared<RuntimeAttribute>();
            runtime_attr->type = RuntimeDataType::kTypeFloat32;
            runtime_attr->weight_data = attr.data;
            runtime_attr->shape = attr.shape;
            runtime_operator->attrs.insert({name, runtime_attr});
            break;
        }
        default: {
            LOG(FATAL) << "Init Graph error, unsupported attribute type: " << attr.type;
            break;
        }
        }
    }
}

void RuntimeGraph::init_op_params(const std::map<std::string, pnnx::Parameter> &params,
                                  const std::shared_ptr<RuntimeOperator> &runtime_operator)
{
    for (const auto &[name, param] : params) {
        if (name.empty()) {
            continue;
        }

        const int type = param.type;
        switch (type) {
        case int(RuntimeParameterType::kParameterUnknown): {
            std::shared_ptr<RuntimeParameter> runtime_parameter = std::make_shared<RuntimeParameter>();
            runtime_operator->params.insert({name, runtime_parameter});
            break;
        }

        case int(RuntimeParameterType::kParameterBool): {
            std::shared_ptr<RuntimeParameterBool> runtime_parameter = std::make_shared<RuntimeParameterBool>();
            runtime_parameter->value = param.b;
            runtime_operator->params.insert({name, runtime_parameter});
            break;
        }

        case int(RuntimeParameterType::kParameterInt): {
            std::shared_ptr<RuntimeParameterInt> runtime_parameter = std::make_shared<RuntimeParameterInt>();
            runtime_parameter->value = param.i;
            runtime_operator->params.insert({name, runtime_parameter});
            break;
        }

        case int(RuntimeParameterType::kParameterFloat): {
            std::shared_ptr<RuntimeParameterFloat> runtime_parameter =
                std::make_shared<RuntimeParameterFloat>();
            runtime_parameter->value = param.f;
            runtime_operator->params.insert({name, runtime_parameter});
            break;
        }

        case int(RuntimeParameterType::kParameterString): {
            std::shared_ptr<RuntimeParameterString> runtime_parameter =
                std::make_shared<RuntimeParameterString>();
            runtime_parameter->value = param.s;
            runtime_operator->params.insert({name, runtime_parameter});
            break;
        }

        case int(RuntimeParameterType::kParameterIntArray): {
            std::shared_ptr<RuntimeParameterIntArray> runtime_parameter =
                std::make_shared<RuntimeParameterIntArray>();
            runtime_parameter->value = param.ai;
            runtime_operator->params.insert({name, runtime_parameter});
            break;
        }

        case int(RuntimeParameterType::kParameterFloatArray): {
            std::shared_ptr<RuntimeParameterFloatArray> runtime_parameter =
                std::make_shared<RuntimeParameterFloatArray>();
            runtime_parameter->value = param.af;
            runtime_operator->params.insert({name, runtime_parameter});
            break;
        }

        case int(RuntimeParameterType::kParameterStringArray): {
            std::shared_ptr<RuntimeParameterStringArray> runtime_parameter =
                std::make_shared<RuntimeParameterStringArray>();
            runtime_parameter->value = param.as;
            runtime_operator->params.insert({name, runtime_parameter});
            break;
        }

        default: {
            LOG(FATAL) << "Unknown parameter type: " << type;
            break;
        }
        }
    }
}

void RuntimeGraph::reverse_topo(const std::shared_ptr<RuntimeOperator> &cur)
{
    CHECK(cur != nullptr) << "start operator is null";
    cur->has_forward = true;

    if (!cur->output_operators.empty()) {
        for (auto &[_, op] : cur->output_operators) {
            if (!op->has_forward) {
                reverse_topo(op);
            }
        }
    }

    this->topo_operators_.push_back(cur);
}

void RuntimeGraph::init_topo_seq(const std::shared_ptr<RuntimeOperator> &start)
{
    this->reverse_topo(start);
    std::reverse(this->topo_operators_.begin(), this->topo_operators_.end());
}

std::shared_ptr<RuntimeOperator> RuntimeGraph::create_op(const std::string& name)
{
    auto runtime_operator = std::make_shared<RuntimeOperator>();
    this->operators_.push_back(runtime_operator);
    this->operators_map_.insert({name, runtime_operator});
    return runtime_operator;
}

GraphState RuntimeGraph::state() const
{
    return this->graph_state_;
}

bool RuntimeGraph::build(std::string input_op_name, std::string output_op_name)
{
    switch (this->graph_state_) {
    case GraphState::need_init:
        LOG(FATAL) << "the graph has not been inited, build fail";
        return false;

    case GraphState::need_build:
        LOG(INFO) << "start build";
        break;

    case GraphState::completed:
        LOG(INFO) << "the graph has been built";
        return true;

    default:
        LOG(FATAL) << "unknown graph state: " << int(this->graph_state_);
        return false;
    }

    this->input_name_ = std::move(input_op_name);
    this->output_name_ = std::move(output_op_name);

    try {
        std::shared_ptr<RuntimeOperator> input_op = this->operators_map_.at(this->input_name_);
        this->topo_operators_.clear();
        this->init_topo_seq(input_op);
    } catch (std::exception &e) {
        LOG(FATAL) << "init topology sequence fail: " << e.what();
        return false;
    }

    this->graph_state_ = GraphState::completed;
    return true;
}

const std::vector<std::shared_ptr<RuntimeOperator>> &RuntimeGraph::get_topo_seq() const
{
    return this->topo_operators_;
}

void RuntimeGraph::check_shape(const std::vector<int> &shape) const
{
    CHECK(shape.size() >= 2 && shape.size() <= 4)
        << "unsupported tensor shape size: " << shape.size();

    auto batch = shape.at(0);
    CHECK(batch >= 0)
        << "dynamic batch size is not supported";
}

void RuntimeGraph::init_data(std::vector<std::shared_ptr<Tensor<float>>> &data, const std::vector<int> &shape)
{
    CHECK(shape.size() >= 2 && shape.size() <= 4)
        << "unsupported shape size: " << shape.size();

    auto batch = shape[0];
    data.resize(batch);

    for (int i = 0; i < batch; i++) {
        switch (shape.size()) {
        case 2:
            data[i] = std::make_shared<ftensor>(shape[1]);
            break;
        case 3:
            data[i] = std::make_shared<ftensor>(shape[1], shape[2]);
            break;
        case 4:
            data[i] = std::make_shared<ftensor>(shape[1], shape[2], shape[3]);
            break;
        }
    }
}

}// namespace jinfer