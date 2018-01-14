#include <cmath>
#include "InplaceOPs.h"
#include "Tools.h"

namespace MiniDL
{
	////////////////////////////
	//functions of SigmoidOP
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
			for (size_t j = 0; j < input_data.size(); j++)
			{
				output_data[j] = 1.0f / (1.0f + std::exp(-input_data[j]));
			}
		}
	}

	//backward of op
	void SigmoidOP::backward(std::vector<Data*>& inputs,
		const std::vector<Data*>& outputs)
	{
		//forward: y = 1.0/(1.0 + e^(-x))
		//backward: dy/dx = y * (1-y)
		//refrence: http://blog.csdn.net/caimouse/article/details/66473030
		do_assert(inputs.size() == outputs.size(), "size of inputs must be equals with outputs.");
		do_assert(inputs.size() > 0, "size of inputs must be larger then zero.");

		// 1.0/(1.0 + e^(-x))
		for (size_t i = 0; i < inputs.size(); i++)
		{
			Data& input_item = *inputs[i];
			const Data& output_item = *outputs[i];
			do_assert(input_item.get_shape() == output_item.get_shape(),
				"size of input must be equals with output. index:%d/%d.", i, (int)inputs.size());
			DataType& input_data = input_item.get_data();
			const DataType& output_data = output_item.get_data();
			for (size_t j = 0; j < input_data.size(); j++)
			{
				input_data[j] = output_data[j] * (1.0f - output_data[j]);
			}
		}
	}

} //namespace MiniDL