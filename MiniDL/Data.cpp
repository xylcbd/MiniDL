#include <algorithm>
#include "Data.h"


namespace MiniDL
{
	Data::Data()
	{

	}

	Data::Data(const Shape& new_shape)
	{
		reshape(new_shape);
	}

	Data::~Data()
	{

	}

	bool Data::is_empty() const
	{
		return shape.get_total() == 0;
	}

	void Data::clear()
	{
		shape.clear();
		mems.clear();
	}

	void Data::reshape(const Shape& new_shape)
	{
		shape = new_shape;
		mems.resize(shape.get_total());
		//fill zero ?
		fill(0.0f);
	}

	Shape Data::get_shape() const
	{
		return shape;
	}

	const DataType& Data::get_data() const
	{
		return mems;
	}

	DataType& Data::get_data()
	{
		return mems;
	}

	void Data::fill(const float val)
	{
		std::fill(mems.begin(), mems.end(), val);
	}
} //namespace MiniDL
