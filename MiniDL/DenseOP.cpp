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
		if (get_phase() == Phase::Train)
		{
			weight_grad.reshape(weight.get_shape());
		}

		// (output_dim)
		if (with_bias)
		{
			bias.reshape(Shape({ output_dim }));
			if (get_phase() == Phase::Train)
			{
				bias_grad.reshape(bias.get_shape());
			}
		}
		else
		{
			bias.clear();
		}

		// init weights
		uniform_filler(weight.get_data(), -1.0f, 1.0f);
		// init bias
		constant_filler(bias.get_data(), 0.0f);

		//init grads
		if (get_phase() == Phase::Train)
		{
			constant_filler(weight_grad.get_data(), 0.0f);
			constant_filler(bias_grad.get_data(), 0.0f);
		}
	}

	const Data& DenseOP::get_weight() const
	{
		return weight;
	}

	const Data& DenseOP::get_bias() const
	{
		return bias;
	}

	//get all weights of op
	std::vector<Data*> DenseOP::get_all_weights()
	{
		std::vector<Data*> all_weights;
		all_weights.push_back(&weight);
		all_weights.push_back(&bias);
		return all_weights;
	}

	//get all grads of op
	std::vector<Data*> DenseOP::get_all_grads()
	{
		std::vector<Data*> all_grads;
		all_grads.push_back(&weight_grad);
		all_grads.push_back(&bias_grad);
		return all_grads;
	}

	//forward of op
	// y = w，x + b
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

			// y = w，x + b
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
	// forawrd: y = w，x + b
	// backward: 
	void DenseOP::backward(const std::vector<Data*>& inputs,
		std::vector<Data*>& prev_diffs,
		const std::vector<Data*>& next_diffs,
		const std::vector<Data*>& outputs)
	{
		//forward: y = w，x + b
		//backward: dy/dx = w, dy/dw = x, dy/db = 1
		//refrence: http://blog.csdn.net/caimouse/article/details/66473030
		do_assert(inputs.size() == outputs.size(), "size of inputs must be equals with outputs.");
		do_assert(prev_diffs.size() == inputs.size(), "size of prev diffs must be equals with inputs.");
		do_assert(next_diffs.size() == outputs.size(), "size of next diffs must be equals with outputs.");
		do_assert(inputs.size() > 0, "size of inputs must be larger then zero.");

		// 
		for (size_t i = 0; i < prev_diffs.size(); i++)
		{
			const Data& input = *inputs[i];
			Data& prev_diff = *prev_diffs[i];
			const Data& next_diff = *next_diffs[i];
			const Data& output = *outputs[i];

			const Shape& input_shape = input.get_shape();
			const Shape& output_shape = output.get_shape();
			const Shape& weight_shape = weight.get_shape();
			const Shape& bias_shape = bias.get_shape();
			const Shape& weight_grad_shape = weight_grad.get_shape();
			const Shape& bias_grad_shape = bias_grad.get_shape();
			const Shape& prev_diff_shape = prev_diff.get_shape();
			const Shape& next_diff_shape = next_diff.get_shape();

			const int input_dim = input_shape.get_dim(1);
			const int diff_dim = input_dim;
			const int output_dim = output_shape.get_dim(1);

			// shape of input: N * input_dim
			// shape of output: N * output_dim
			// shape of weight: input_dim * output_dim
			// shape of bias: output_dim
			do_assert(input_shape.get_dims() == output_shape.get_dims() &&
				input_shape.get_dim(0) == output_shape.get_dim(0) &&
				input_dim == weight_shape.get_dim(0) &&
				output_dim == weight_shape.get_dim(1) &&
				weight_grad_shape == weight_shape &&
				bias_grad_shape == bias_shape &&
				prev_diff_shape == input_shape &&
				next_diff_shape == output_shape &&
				(bias.is_empty() ? true : output_dim == bias_shape.get_dim(0)),
				"dim of input or output is invalidate");

			const DataType& input_mems = input.get_data();			
			DataType& prev_diff_mems = prev_diff.get_data();			
			const DataType& next_diff_mems = next_diff.get_data();
			const DataType& output_mems = output.get_data();
			const DataType& weight_mems = weight.get_data();
			const DataType& bias_mems = bias.get_data();			
			DataType& weight_grad_mems = weight_grad.get_data();			
			DataType& bias_grad_mems = bias_grad.get_data();

			const int batch_size = input_shape.get_dim(0);
			//dy/dx = w --> diff
			prev_diff.fill(0.0f);
			for (size_t i = 0; i < batch_size; i++)
			{
				for (size_t j = 0; j < input_dim; j++)
				{
					for (size_t k = 0; k < output_dim; k++)
					{
						const size_t x_idx = i * input_dim + j;
						const float x = input_mems[x_idx];
						const size_t y_idx = i * output_dim + k;
						const float y = output_mems[y_idx];
						const float w = weight_mems[j * output_dim + k];

						const float nex_diff_val = next_diff_mems[y_idx];
						prev_diff_mems[x_idx] += w * nex_diff_val;
					}
				}
			}

			//dy/dw = x --> weight_grad
			weight_grad.fill(0.0f);
			for (size_t i = 0; i < batch_size; i++)
			{
				for (size_t j = 0; j < input_dim; j++)
				{
					for (size_t k = 0; k < output_dim; k++)
					{
						const float x = input_mems[i * input_dim + j];
						const size_t y_idx = i * output_dim + k;
						const float y = output_mems[y_idx];
						const float nex_diff_val = next_diff_mems[y_idx];

						const size_t w_idx = j * output_dim + k;
						const float w = weight_mems[w_idx];
						weight_grad_mems[w_idx] += x* nex_diff_val;
					}					
				}
			}
			for (size_t i = 0; i < weight_grad.get_shape().get_total(); i++)
			{
				weight_grad_mems[i] /= batch_size;
			}

			//dy/db = 1 --> bias_grad
			bias_grad.fill(0.0f);
			if (!bias.is_empty())
			{
				for (size_t i = 0; i < batch_size; i++)
				{
					for (size_t k = 0; k < output_dim; k++)
					{
						const size_t b_idx = k;
						const size_t y_idx = i * output_dim + k;
						const float nex_diff_val = next_diff_mems[y_idx];

						bias_grad_mems[b_idx] += 1.0f * nex_diff_val;
					}
				}
				for (size_t i = 0; i < bias_grad.get_shape().get_total(); i++)
				{
					bias_grad_mems[i] /= batch_size;
				}
			}
		}
	};

} // namespace MiniDL