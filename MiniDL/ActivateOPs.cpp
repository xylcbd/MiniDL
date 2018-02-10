#include <cmath>
#include "ActivateOPs.h"
#include "Tools.h"

namespace MiniDL
{
	////////////////////////////
	//functions of SigmoidOP

	//get all weights of op
	std::vector<Data*> SigmoidOP::get_all_weights()
	{
		std::vector<Data*> all_weights;
		return all_weights;
	}

	//get all grads of op
	std::vector<Data*> SigmoidOP::get_all_grads()
	{
		std::vector<Data*> all_grads;
		return all_grads;
	}

	//forward of op
	void SigmoidOP::forward(const std::vector<Data*>& inputs,
		std::vector<Data*>& outputs)
	{	
		do_assert(inputs.size() == outputs.size(), "size of inputs must be equals with outputs.");
		do_assert(inputs.size() > 0, "size of inputs must be larger then zero.");

		// 1.0/(1.0 + e^(-x))
		for (size_t i = 0; i < inputs.size(); i++)
		{
			const Data& input_item = *inputs[i];
			Data& output_item = *outputs[i];
			do_assert(input_item.get_shape() == output_item.get_shape(), 
				"size of input must be equals with output. index:%d/%d.", i, (int)inputs.size());
			const DataType& input_data = input_item.get_data();
			DataType& output_data = output_item.get_data();
			//已考虑batch_size
			for (size_t j = 0; j < input_data.size(); j++)
			{
				output_data[j] = 1.0f / (1.0f + std::exp(-input_data[j]));
			}
		}
	}

	//backward of op
	void SigmoidOP::backward(const std::vector<Data*>& inputs,
		std::vector<Data*>& prev_diffs,
		const std::vector<Data*>& next_diffs,
		const std::vector<Data*>& outputs)
	{
		//forward: y = 1.0/(1.0 + e^(-x))
		//backward: dy/dx = y * (1-y)
		//refrence: http://blog.csdn.net/caimouse/article/details/66473030
		do_assert(inputs.size() == outputs.size(), "size of inputs must be equals with outputs.");
		do_assert(prev_diffs.size() == inputs.size(), "size of prev diffs must be equals with inputs.");
		do_assert(next_diffs.size() == outputs.size(), "size of next diffs must be equals with outputs.");
		do_assert(inputs.size() > 0, "size of inputs must be larger then zero.");

		// 1.0/(1.0 + e^(-x))
		for (size_t i = 0; i < prev_diffs.size(); i++)
		{
			Data& prev_diff = *prev_diffs[i];
			const Data& next_diff = *next_diffs[i];
			const Data& input = *inputs[i];
			const Data& output = *outputs[i];
			do_assert(prev_diff.get_shape() == input.get_shape(),
				"size of prev diff must be equals with input. index:%d/%d.", i, (int)prev_diffs.size());
			do_assert(next_diff.get_shape() == output.get_shape(),
				"size of next diff must be equals with output. index:%d/%d.", i, (int)next_diffs.size());

			const DataType& input_mems = input.get_data();
			prev_diff.fill(0.0f);
			DataType& prev_diff_mems = prev_diff.get_data();
			const DataType& next_diff_mems = next_diff.get_data();
			const DataType& output_mems = output.get_data();
			//已考虑batch_size
			// 计算当前产生的残差
			for (size_t j = 0; j < prev_diff_mems.size(); j++)
			{
				prev_diff_mems[j] = output_mems[j] * (1.0f - output_mems[j]);
			}
			for (size_t j = 0; j < prev_diff_mems.size(); j++)
			{
				prev_diff_mems[j] *= next_diff_mems[j];
			}
		}
	}

} //namespace MiniDL