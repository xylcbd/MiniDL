#include "InputOP.h"
#include "Tools.h"

namespace MiniDL
{
	InputOP::InputOP()
	{

	}
	InputOP::~InputOP()
	{

	}
	//get all weights of op
	std::vector<Data*> InputOP::get_all_weights()
	{
		std::vector<Data*> outputs;
		return outputs;
	}

	//get all grads of op
	std::vector<Data*> InputOP::get_all_grads()
	{
		std::vector<Data*> outputs;
		return outputs;
	}

	//forward of op
	void InputOP::forward(const std::vector<Data*>& inputs,
		std::vector<Data*>& outputs)
	{
		do_assert(inputs.size() == outputs.size(), "size of inputs is not equals with outpus");

		for (size_t i = 0; i < inputs.size(); i++)
		{
			const Data& input = *inputs[i];
			Data& output = *outputs[i];

			const Shape& input_shape = input.get_shape();
			const Shape& output_shape = output.get_shape();

			const int input_dim = input_shape.get_dim(1);
			const int output_dim = output_shape.get_dim(1);

			// shape of input: N * input_dim
			// shape of output: N * output_dim			
			do_assert(input_shape == output_shape, "shape of input must be equals with output");

			// y = w¡¤x + b
			const DataType& input_mems = input.get_data();
			DataType& output_mems = output.get_data();
			output_mems = input_mems;
		}
	}

	//backward of op
	void InputOP::backward(const std::vector<Data*>& inputs,
		std::vector<Data*>& prev_diffs,
		const std::vector<Data*>& next_diffs,
		const std::vector<Data*>& outputs)
	{
		//NO operater
	}
} //namespace MiniDL