#pragma once
#include <vector>
#include <initializer_list>

namespace MiniDL
{
	class Shape
	{
	public:		
		Shape();
		Shape(const std::initializer_list<int> items);
		~Shape();
	public:
		bool operator==(const Shape& other) const;
		int get_total() const;
		int get_dims() const;
		int get_dim(const int index) const;
	private:
		std::vector<int> data;
	};

}// namespace MiniDL