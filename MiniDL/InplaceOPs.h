#pragma once
#include "Operator.h"

namespace MiniDL
{
	//active functions

	//sigmoid function
	// 1.0/(1.0 + e^(-x))
	class SigmoidOP : public Operator
	{
	public:
		//forward of op
		virtual void forward(const std::vector<Data*>& inputs,
			std::vector<Data*>& outputs);

		//backward of op
		virtual void backward(std::vector<Data*>& inputs,
			const std::vector<Data*>& outputs);
	};


} //namespace MiniDL