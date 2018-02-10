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
		enum class Phase
		{
			Train,
			Test
		};
		inline void set_phase(const Phase phase)
		{
			this->phase = phase;
		}
		inline Phase get_phase() const
		{
			return this->phase;
		}

		//get all weights of op
		virtual std::vector<Data*> get_all_weights() = 0;

		//get all grads of op
		virtual std::vector<Data*> get_all_grads() = 0;

		//forward of op
		virtual void forward(const std::vector<Data*>& inputs,
			std::vector<Data*>& outputs) = 0;

		//backward of op
		virtual void backward(const std::vector<Data*>& inputs,
			std::vector<Data*>& prev_diffs,
			const std::vector<Data*>& next_diffs,
			const std::vector<Data*>& outputs) = 0;

	private:
		Phase phase = Phase::Test;
	};
} //namespace MiniDL