#include "Tools.h"
#include <cstdarg>
#include <cstdio>
#include <iostream>

namespace MiniDL
{
	//custom assert
	void do_assert(const bool condition, const char* fmt, ...)
	{
		if (!condition)
		{
			//log to console.
			va_list ap;
			va_start(ap, fmt);
			fprintf(stderr, fmt, ap);
			va_end(ap);
		}
	}

	//log shape
	void log_shape(const Shape& shape)
	{
		//log to console
		//details
		std::cout << "shape: (";
		for (int i = 0; i < shape.get_dims(); i++)
		{
			std::cout << shape.get_dim(i) << " ";
		}
		std::cout << ")" << std::endl;
	}

	//log data
	void log_data(const Data& data)
	{
		//log to console
		log_shape(data.get_shape());
		//details
		std::cout << "data: [";
		const DataType& details = data.get_data();
		for (size_t i = 0; i < details.size(); i++)
		{
			std::cout << details[i] << " ";
		}
		std::cout << "]" << std::endl;
	}
}// namespace MiniDL