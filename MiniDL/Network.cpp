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

	// 设置LossFunctor
	void Network::set_loss_functor(LossFunctor* loss_functor)
	{
		do_assert(loss_functor != nullptr, "loss functor can't be NULL.");
		this->loss_functor = loss_functor;
	}

	// 前向传播
	void Network::forward(const std::vector<Data*>& inputs, std::vector<Data*>& predicts)
	{
		//TODO
	}

	// 反向传播
	void Network::backward(const std::vector<Data*>& inputs, const std::vector<Data*>& groundtruthes)
	{
		do_assert(loss_functor != nullptr, "loss functor can't be NULL.");
		//TODO
	}

	// loss计算
	float Network::get_loss(const std::vector<Data*>& groundtruthes, const std::vector<Data*>& predicts)
	{
		do_assert(loss_functor != nullptr, "loss functor can't be NULL.");
		return loss_functor->get_loss(groundtruthes, predicts);
	}

	// 更新参数
	void Network::update_weights()
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