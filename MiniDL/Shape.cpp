#include "Shape.h"
#include "Tools.h"

namespace MiniDL
{
	Shape::Shape()
	{
	}

	Shape::Shape(const std::initializer_list<int> items)
	{
		mems = items;
	}

	Shape::~Shape()
	{
	}

	int Shape::get_total() const
	{
		if (mems.empty())
		{
			return 0;
		}
		int acc = 1;
		for (size_t i = 0; i < mems.size(); i++)
		{
			acc *= mems[i];
		}		
		return acc;
	}

	int Shape::get_dims() const
	{
		return (int)mems.size();
	}

	int Shape::get_dim(const int index) const
	{
		do_assert(index >= 0 && index < mems.size(), "index is invalidate.");
		return mems[index];
	}

	void Shape::clear()
	{
		mems.clear();
	}

	bool Shape::operator==(const Shape& other) const
	{
		return this->mems == other.mems;
	}
} //namespace MiniDL