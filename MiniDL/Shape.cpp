#include "Shape.h"
#include "Tools.h"

namespace MiniDL
{
	Shape::Shape()
	{
	}

	Shape::Shape(const std::initializer_list<int> items)
	{
		data = items;
	}

	Shape::~Shape()
	{
	}

	int Shape::get_total() const
	{
		int acc = 1;
		for (size_t i = 0; i < data.size(); i++)
		{
			acc *= data[i];
		}
		return acc;
	}

	int Shape::get_dims() const
	{
		return (int)data.size();
	}

	int Shape::get_dim(const int index) const
	{
		do_assert(index >= 0 && index < data.size(), "index is invalidate.");
		return data[index];
	}

	bool Shape::operator==(const Shape& other) const
	{
		return this->data == other.data;
	}
} //namespace MiniDL