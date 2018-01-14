#pragma once
#include "Configure.h"

namespace MiniDL
{
	//interface of loss function
	interface LossFunctor
	{
	public:
		float get_loss();
	};
} //namespace MiniDL
