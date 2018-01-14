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

	void Data::reshape(const Shape& new_shape)
	{
		shape = new_shape;
		data.resize(shape.get_total());
		//fill zero ?
		fill(0.0f);
	}

	Shape Data::get_shape() const
	{
		return shape;
	}

	const DataType& Data::get_data() const
	{
		return data;
	}

	DataType& Data::get_data()
	{
		return data;
	}

	void Data::fill(const float val)
	{
		std::fill(data.begin(), data.end(), val);
	}
} //namespace MiniDL
