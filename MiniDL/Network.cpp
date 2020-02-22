#pragma once
#include "Network.h"
#include "Tools.h"

namespace MiniDL
{
	Network& Network::add_op(Operator* op)
	{
		do_assert(op != nullptr, "op can't be NULL.");
		ops.push_back(op);
		return *this;
	}

	// ????LossFunctor
	void Network::set_loss_functor(LossFunctor* loss_functor)
	{
		do_assert(loss_functor != nullptr, "loss functor can't be NULL.");
		this->loss_functor = loss_functor;
	}

	// ǰ?򴫲?
	void Network::forward(const std::vector<Data*>& inputs, std::vector<Data*>& predicts)
	{
		//TODO
	}

	// ???򴫲?
	void Network::backward(const std::vector<Data*>& inputs, const std::vector<Data*>& groundtruthes)
	{
		do_assert(loss_functor != nullptr, "loss functor can't be NULL.");
		//TODO
	}

	// loss????
	float Network::get_loss(const std::vector<Data*>& groundtruthes, const std::vector<Data*>& predicts)
	{
		do_assert(loss_functor != nullptr, "loss functor can't be NULL.");
		return loss_functor->get_loss(groundtruthes, predicts);
	}

	// ???²???
	void Network::update_weights()
	{
		//TODO
	}

	// ????ģ??
	bool Network::save_model(const std::string& model_path) const
	{
		//TODO
		return false;
	}

	// ????ģ??
	bool Network::load_model(const std::string& model_path)
	{
		//TODO
		return false;
	}
} //namespace MiniDL
