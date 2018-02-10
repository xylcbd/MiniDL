#pragma once
#include "network.h"
#include "tools.h"

namespace MiniDL
{
	Network& Network::add_op(Operator* op)
	{
		do_assert(op != nullptr, "op can't be NULL.");
		ops.push_back(op);
		return *this;
	}

	// 前向传播
	void Network::forward(const std::vector<Data*>& inputs)
	{
		//TODO
	}

	// 反向传播
	void Network::backward(const std::vector<Data*>& inputs, const std::vector<Data*>& groundtruthes)
	{
		//TODO
	}

	// 保存模型
	bool Network::save_model(const std::string& model_path) const
	{
		//TODO
		return false;
	}

	// 加载模型
	bool Network::load_model(const std::string& model_path)
	{
		//TODO
		return false;
	}
} //namespace MiniDL