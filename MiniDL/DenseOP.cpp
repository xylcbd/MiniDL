#include "DenseOP.h"
#include "MathFunctions.h"
#include "Tools.h"

namespace MiniDL
{
	DenseOP::DenseOP()
	{

	}
	DenseOP::~DenseOP()
	{

	}

	// setting
	void DenseOP::setup(const int input_dim, const int output_dim, const bool with_bias)
	{
		// (input_dim, output_dim)
		weight.reshape(Shape({ input_dim, output_dim }));
		// (output_dim)
		if (with_bias)
		{
			bias.reshape(Shape({ output_dim }));
		}
		else
		{
			bias.clear();
		}

		// init weights
		uniform_filler(weight.get_data(), -1.0f, 1.0f);
		// init bias
		constant_filler(bias.get_data(), 0.0f);
	}

	const Data& DenseOP::get_weight() const
	{
		return weight;
	}

	const Data& DenseOP::get_bias() const
	{
		return bias;
	}

	//forward of op
	// y = w¡¤x + b
	void DenseOP::forward(const std::vector<Data*>& inputs,
		std::vector<Data*>& outputs)
	{
		do_assert(inputs.size() == outputs.size(), "size of inputs is not equals with outpus");
		
		for (size_t i = 0; i < inputs.size(); i++)
		{
			const Data& input = *inputs[i];			
			Data& output = *outputs[i];

			const Shape& input_shape = input.get_shape();
			const Shape& output_shape = output.get_shape();
			const Shape& weight_shape = weight.get_shape();
			const Shape& bias_shape = bias.get_shape();

			const int input_dim = input_shape.get_dim(1);
			const int output_dim = output_shape.get_dim(1);

			// shape of input: N * input_dim
			// shape of output: N * output_dim
			// shape of weight: input_dim * output_dim
			// shape of bias: output_dim
			do_assert(input_shape.get_dims() == output_shape.get_dims() &&
				input_shape.get_dim(0) == output_shape.get_dim(0) &&
				input_dim == weight_shape.get_dim(0) &&
				output_dim == weight_shape.get_dim(1) &&
				(bias.is_empty() ? true : output_dim == bias_shape.get_dim(0)),
				"dim of input or output is invalidate");

			// y = w¡¤x + b
			const DataType& input_mems = input.get_data();
			DataType& output_mems = output.get_data();
			const DataType& weight_mems = weight.get_data();
			const DataType& bias_mems = bias.get_data();
			const int batch_size = input_shape.get_dim(0);
			for (size_t i = 0; i < batch_size; i++)
			{
				// output_dim
				// [y0 y1...yL] ... [y0 y1...yL]
				for (size_t j = 0; j < output_dim; j++)
				{
					float sum = 0.0f;
					for (size_t k = 0; k < input_dim; k++)
					{
						sum += weight_mems[k*output_dim + j] * input_mems[i*input_dim + k];
					}
					if (!bias.is_empty())
					{
						sum += bias_mems[j];
					}
					output_mems[i * output_dim + j] = sum;
				}
			}
		}
	}

	//backward of op
	void DenseOP::backward(std::vector<Data*>& inputs,
		const std::vector<Data*>& outputs)
	{
		//TODO
	};

} // namespace MiniDL