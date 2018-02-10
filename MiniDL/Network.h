#pragma once
#pragma once
#include "Configure.h"
#include "Data.h"
#include "Operator.h"

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

		// 前向传播
		void forward(const std::vector<Data*>& inputs);

		// 反向传播
		void backward(const std::vector<Data*>& inputs, const std::vector<Data*>& groundtruthes);

		// 保存模型
		bool save_model(const std::string& model_path) const;

		// 加载模型
		bool load_model(const std::string& model_path);
	private:
		std::vector<Operator*> ops;
	};

} //namespace MiniDL