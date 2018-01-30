#pragma once
#include "Configure.h"
#include "Shape.h"

namespace MiniDL
{
	class Data
	{
	public:
		Data();
		Data(const Shape& new_shape);
		~Data();
	public:
		bool is_empty() const;
		void clear();
		void reshape(const Shape& new_shape);
		Shape get_shape() const;
		const DataType& get_data() const;
		DataType& get_data();
		void fill(const float val);
	private:
		//shape of data
		Shape shape;
		//data container
		DataType mems;
	};
} //namespace MiniDL
