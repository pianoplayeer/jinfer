//
// Created by 27836 on 2025/6/6.
//

#include "data/tensor.hpp"
#include <glog/logging.h>

namespace jinfer
{

Tensor<float>::Tensor(uint32_t size)
{
	data_ = arma::fcube(1, size, 1);
	raw_shape_ = std::vector<uint32_t>{ size };
}

Tensor<float>::Tensor(uint32_t rows, uint32_t cols)
{
	data_ = arma::fcube(rows, cols, 1);

	if (rows == 1) {
		raw_shape_ = std::vector<uint32_t>{ cols };
	} else {
		raw_shape_ = std::vector<uint32_t>{ rows, cols };
	}
}

Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols)
{
	data_ = arma::fcube(rows, cols, channels);
	if (channels == 1 && rows == 1) {
		raw_shape_ = std::vector<uint32_t>{ cols };
	} else if (channels == 1) {
		raw_shape_ = std::vector<uint32_t>{ rows, cols };
	} else {
		raw_shape_ = std::vector<uint32_t>{ channels, rows, cols };
	}
}

Tensor<float>::Tensor(const std::vector<uint32_t> &shapes)
{
	CHECK(!shapes.empty() && shapes.size() <= 3);

	uint32_t remaining = 3 - shapes.size();
	std::vector<uint32_t> shapes_(3, 1);
	std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

	uint32_t channels = shapes_.at(0);
	uint32_t rows = shapes_.at(1);
	uint32_t cols = shapes_.at(2);

	data_ = arma::fcube(rows, cols, channels);
	if (channels == 1 && rows == 1) {
		raw_shape_ = std::vector<uint32_t>{ cols };
	} else if (channels == 1) {
		raw_shape_ = std::vector<uint32_t>{ rows, cols };
	} else {
		raw_shape_ = std::vector<uint32_t>{ channels, rows, cols };
	}
}

Tensor<float>::Tensor(const Tensor &tensor)
{
	if (this != &tensor) {
		this->data_ = tensor.data_;
		this->raw_shape_ = tensor.raw_shape_;
	}
}

Tensor<float>::Tensor(Tensor &&tensor) noexcept
{
	if (this != &tensor) {
		this->data_ = std::move(tensor.data_);
		this->raw_shape_ = tensor.raw_shape_;
	}
}

Tensor<float> &Tensor<float>::operator=(Tensor &&tensor) noexcept
{
	if (this != &tensor) {
		this->data_ = std::move(tensor.data_);
		this->raw_shape_ = tensor.raw_shape_;
	}

	return *this;
}

Tensor<float> &Tensor<float>::operator=(const Tensor &tensor)
{
	if (this != &tensor) {
		this->data_ = tensor.data_;
		this->raw_shape_ = tensor.raw_shape_;
	}

	return *this;
}

uint32_t
Tensor<float>::rows() const
{
	CHECK(!this->data_.empty());
	return data_.n_rows;
}

uint32_t
Tensor<float>::cols() const
{
	CHECK(!this->data_.empty());
	return data_.n_cols;
}

uint32_t
Tensor<float>::channels() const
{
	CHECK(!this->data_.empty());
	return data_.n_slices;
}

uint32_t
Tensor<float>::size() const
{
	CHECK(!this->data_.empty());
	return data_.size();
}

void
Tensor<float>::set_data(const arma::fcube &data)
{
	CHECK(this->data_.n_rows == data.n_rows)
			<< this->data_.n_rows << " != " << data.n_rows;
	CHECK(this->data_.n_cols == data.n_cols)
			<< this->data_.n_cols << " != " << data.n_cols;
	CHECK(this->data_.n_slices == data.n_slices)
			<< this->data_.n_slices << " != " << data.n_slices;

	this->data_ = data;
}

const std::vector<uint32_t> &
Tensor<float>::raw_shapes() const
{
	CHECK(!this->raw_shape_.empty());
	CHECK_LE(this->raw_shape_.size(), 3);
	CHECK_GE(this->raw_shape_.size(), 1);

	return this->raw_shape_;
}

void
Tensor<float>::fill(float value)
{
	CHECK(!this->data_.empty());
	this->data_.fill(value);
}

void
Tensor<float>::show()
{
	for (uint32_t i = 0; i < this->channels(); i++) {
		LOG(INFO) << "channel: " << i;
		LOG(INFO) << "\n" << this->data_.slice(i);
	}
}

const arma::fmat &
Tensor<float>::slice(uint32_t channel) const
{
	CHECK_LT(channel, this->channels());
	return this->data_.slice(channel);
}

float
Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col)
{
	CHECK_LT(row, this->rows());
	CHECK_LT(col, this->cols());
	CHECK_LT(channel, this->channels());

	return this->data_.at(row, col, channel);
}

void
Tensor<float>::rand()
{
	CHECK(!this->data_.empty());
	this->data_.randn();
}

void
Tensor<float>::ones()
{
	CHECK(!this->data_.empty());
	this->fill(1);
}

void
Tensor<float>::reshape(const std::vector<uint32_t> &shapes, bool row_major)
{
	CHECK(!this->data_.empty());
	CHECK(!shapes.empty());

	uint32_t origin_size = this->size();
	uint32_t current_size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
	CHECK(shapes.size() <= 3);
	CHECK(origin_size == current_size);

	uint32_t rows = 1, cols = 1, channels = 1;
	assign_each_shape(shapes, rows, cols, channels);

	this->data_ = arma::fcube(rows, cols, channels);
	if (channels > 1) {
		this->raw_shape_ = { channels, rows, cols };
	} else if (rows > 1) {
		this->raw_shape_ = { rows, cols };
	} else {
		this->raw_shape_ = { cols };
	}

	std::vector<float> values = this->values(row_major);
	this->fill(values, row_major);
}

std::vector<float>
Tensor<float>::values(bool row_major)
{
	CHECK(!this->data_.empty());
	uint32_t size = this->size();
	std::vector<float> values(size);

	if (row_major) {
		uint32_t index = 0;
		for (int i = 0; i < channels(); i++) {
			const arma::fmat &mat = slice(i).t();
			std::copy(mat.begin(), mat.end(), values.begin() + index);
			index += mat.size();
		}
	} else {
		std::copy(this->data_.begin(), this->data_.end(), values.begin());
	}

	return values;
}

void
Tensor<float>::fill(const std::vector<float> &values, bool row_major)
{
	CHECK(!this->data_.empty());
	CHECK(this->data_.size() == values.size());

	if (row_major) {
		uint32_t channels = this->channels();
		uint32_t rows = this->rows();
		uint32_t cols = this->cols();
		uint32_t plane = rows * cols;

		for (int i = 0; i < channels; i++) {
			auto &channel = this->data_.slice(i);
			const arma::fmat &mat = arma::fmat((float*)values.data() + i * plane,
				cols, rows, false, true);
			channel = mat.t();
		}
	} else {
		std::copy(values.begin(), values.end(), this->data_.begin());
	}
}

void Tensor<float>::transform(const std::function<float(float)> &filter)
{
	CHECK(!this->data_.empty());
	this->data_.transform(filter);
}

bool Tensor<float>::empty()
{
	return this->data_.empty();
}

const float* Tensor<float>::raw_ptr() const
{
	CHECK(!this->data_.empty());
	return this->data_.memptr();
}

void Tensor<float>::flatten(bool row_major)
{
	CHECK(!this->data_.empty());
	uint32_t size = this->size();

	this->reshape({1, 1, size}, row_major);
}

void Tensor<float>::padding(const std::vector<uint32_t> &pads, float padding_value)
{
	CHECK(!this->data_.empty());
	CHECK_EQ(pads.size(), 4);

	uint32_t up = pads.at(0);
	uint32_t down = pads.at(1);
	uint32_t left = pads.at(2);
	uint32_t right = pads.at(3);

	uint32_t rows = this->rows() + up + down;
	uint32_t cols = this->cols() + left + right;
	uint32_t channels = this->channels();

	arma::fcube data(rows, cols, channels);
	data.fill(padding_value);

	data.subcube(up, left, 0,
		rows - down - 1, cols - right - 1, channels - 1) = this->data_;
	this->data_ = std::move(data);

	if (channels > 1 && rows > 1) {
		this->raw_shape_ = {channels, rows, cols};
	} else if (rows > 1) {
		this->raw_shape_ = {rows, cols};
	} else {
		this->raw_shape_ = {cols};
	}
}

void Tensor<float>::assign_each_shape(std::vector<uint32_t> shapes, uint32_t &rows, uint32_t &cols, uint32_t &channels)
{
	rows = 1;
	cols = 1;
	channels = 1;

	switch (shapes.size()) {
	case 3:
		channels = shapes.at(0);
		rows = shapes.at(1);
		cols = shapes.at(2);
		break;

	case 2:
		rows = shapes.at(0);
		cols = shapes.at(1);
		break;

	case 1:
		cols = shapes.at(0);
		break;

	default:
		break;
	}
}

}