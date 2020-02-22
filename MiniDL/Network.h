#pragma once
#pragma once
#include<string>
#include "Configure.h"
#include "Data.h"
#include "Operator.h"
#include "LossFunctor.h"

namespace MiniDL
{
	// 1. ?????????????˽ṹ
	// 2. ???????ݵ?????
	// 3. ģ?ͼ??غͱ???	
	class Network
	{
	public:
		// ????OP
		Network& add_op(Operator* op);

		// ????LossFunctor
		void set_loss_functor(LossFunctor* loss_functor);

		// ǰ?򴫲?
		void forward(const std::vector<Data*>& inputs, std::vector<Data*>& predicts);

		// ???򴫲?
		void backward(const std::vector<Data*>& inputs, const std::vector<Data*>& groundtruthes);

		// loss????
		float get_loss(const std::vector<Data*>& groundtruthes, const std::vector<Data*>& predicts);

		// ???²???
		void update_weights();

		// ????ģ??
		bool save_model(const std::string& model_path) const;

		// ????ģ??
		bool load_model(const std::string& model_path);
	private:
		// ????op
		std::vector<Operator*> ops;
		// loss_functor
		LossFunctor* loss_functor = nullptr;
	};

} //namespace MiniDL
