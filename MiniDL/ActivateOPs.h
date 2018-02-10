#pragma once
#include "Operator.h"

namespace MiniDL
{
	//active functions

	//sigmoid function
	// y = 1.0/(1.0 + e^(-x))
	class SigmoidOP : public Operator
	{
	public:
		//get all weights of op
		virtual std::vector<Data*> get_all_weights();

		//get all grads of op
		virtual std::vector<Data*> get_all_grads();

		//forward of op
		virtual void forward(const std::vector<Data*>& inputs,
			std::vector<Data*>& outputs);

		//backward of op
		virtual void backward(const std::vector<Data*>& inputs,
			std::vector<Data*>& prev_diffs,
			const std::vector<Data*>& next_diffs,
			const std::vector<Data*>& outputs);
	};


} //namespace MiniDL