#pragma once
#pragma once
#include "Configure.h"
#include "Data.h"
#include "Operator.h"
#include "LossFunctor.h"

namespace MiniDL
{
	// 1. 描述网络的拓扑结构
	// 2. 控制数据的流向
	// 3. 模型加载和保存	
	class Network
	{
	public:
		// 增加OP
		Network& add_op(Operator* op);

		// 设置LossFunctor
		void set_loss_functor(LossFunctor* loss_functor);

		// 前向传播
		void forward(const std::vector<Data*>& inputs, std::vector<Data*>& predicts);

		// 反向传播
		void backward(const std::vector<Data*>& inputs, const std::vector<Data*>& groundtruthes);

		// loss计算
		float get_loss(const std::vector<Data*>& groundtruthes, const std::vector<Data*>& predicts);

		// 更新参数
		void update_weights();

		// 保存模型
		bool save_model(const std::string& model_path) const;

		// 加载模型
		bool load_model(const std::string& model_path);
	private:
		// 所有op
		std::vector<Operator*> ops;
		// loss_functor
		LossFunctor* loss_functor = nullptr;
	};

} //namespace MiniDL