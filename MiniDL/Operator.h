#pragma once
#include <vector>
#include "Configure.h"
#include "Data.h"

namespace MiniDL
{
	//base class of OPs
	interface Operator
	{
	public:
		//TODO: add weights, grads etc.

		//forward of op
		virtual void forward(const std::vector<Data*>& inputs,
			std::vector<Data*>& outputs) = 0;

		//backward of op
		virtual void backward(std::vector<Data*>& inputs,
			const std::vector<Data*>& outputs) = 0;
	};
} //namespace MiniDL