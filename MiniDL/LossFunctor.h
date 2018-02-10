#pragma once
#include "Configure.h"
#include "Data.h"

namespace MiniDL
{
	//interface of loss function
	interface LossFunctor
	{
	public:
		virtual float get_loss(const std::vector<Data*>& groundtruthes, const std::vector<Data*>& predicts) = 0;
		virtual std::vector<Data*> get_grads(const std::vector<Data*>& groundtruthes, const std::vector<Data*>& predicts) = 0;
	};

	class MSELoss : public LossFunctor
	{
	public:
		virtual float get_loss(const std::vector<Data*>& groundtruthes, const std::vector<Data*>& predicts);
		virtual std::vector<Data*> get_grads(const std::vector<Data*>& groundtruthes, const std::vector<Data*>& predicts);
	};
} //namespace MiniDL
