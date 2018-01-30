#pragma once
#include "Configure.h"

namespace MiniDL
{
	// 常量初始化
	void constant_filler(DataType& data, const float val);

	// 高斯分布初始化
	void gaussian_filler(DataType& data, const float mean, const float std);

	// 均匀分布初始化
	void uniform_filler(DataType& data, const float min_val, const float max_val);
}