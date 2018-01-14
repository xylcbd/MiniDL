#pragma once

#include "Operator.h"

namespace MiniDL
{
	class DenseOP : public Operator
	{
	public:
		DenseOP();
		virtual ~DenseOP();
	public:
		//forward of op
		virtual void forward(const std::vector<Data*>& inputs,
			std::vector<Data*>& outputs);

		//backward of op
		virtual void backward(std::vector<Data*>& inputs,
			const std::vector<Data*>& outputs);

	private:
		//inner parameters
	};
} //namespace MiniDL