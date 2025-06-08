//
// Created by 27836 on 2025/6/6.
//

#ifndef JINFER_TENSOR_HPP
#define JINFER_TENSOR_HPP

#include <armadillo>
#include <memory>

namespace jinfer
{

template<typename T = float>
class Tensor
{
};

template<>
class Tensor<uint8_t>
{
	// 待实现
};

template<>
class Tensor<float>
{
public:
	explicit Tensor() = default;

	explicit Tensor(uint32_t size);

	explicit Tensor(uint32_t rows, uint32_t cols);

	explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

	explicit Tensor(const std::vector<uint32_t> &shapes);

	Tensor(const Tensor &tensor);

	Tensor(Tensor &&tensor) noexcept;

	Tensor<float> &operator=(Tensor &&tensor) noexcept;

	Tensor<float> &operator=(const Tensor &tensor);

	uint32_t
	rows() const;

	uint32_t
	cols() const;

	uint32_t
	channels() const;

	uint32_t
	size() const;

	const std::vector<uint32_t> &
	raw_shapes() const;

	void
	set_data(const arma::fcube &data);

	void
	fill(float value);

	void
	fill(const std::vector<float> &values, bool row_major = true);

	void
	show();

	const arma::fmat &
	slice(uint32_t channel) const;

	float
	at(uint32_t channel, uint32_t row, uint32_t col);

	void
	rand();

	void
	ones();

	void
	reshape(const std::vector<uint32_t> &shapes, bool row_major = true);

	std::vector<float>
	values(bool row_major = false);

	void
	transform(const std::function<float(float)> &filter);

	bool
	empty();

	const float *
	raw_ptr() const;

	void
	flatten(bool row_major = true);

	/**
	 * @param pads index represents: 0->up, 1->bottom, 2->left, 3->right
	 * @param padding_value
	 */
	void
	padding(const std::vector<uint32_t> &pads, float padding_value = 0);

private:
	std::vector<uint32_t> raw_shape_;
	arma::fcube data_;

	void
	assign_each_shape(std::vector<uint32_t> shapes, uint32_t &rows, uint32_t &cols, uint32_t &channels);
};

using ftensor = Tensor<float>;
using sftensor = std::shared_ptr<Tensor<float>>;

}

#endif //JINFER_TENSOR_HPP
