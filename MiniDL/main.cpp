#include <iostream>
#include <cstdlib>
#include <string>
#include <opencv2/opencv.hpp>
#include "MiniDL.h"

static void test_assert()
{
	MiniDL::do_assert(true, "you can't see this message.");
	MiniDL::do_assert(false, "test only.");
}

static void test_sigmodi()
{
	MiniDL::SigmoidOP sigmoidOP;
	MiniDL::Shape shape({ 2,2 });
	MiniDL::Data input(shape);
	input.fill(0.5f);
	MiniDL::Data output(shape);
	output.fill(0.0f);
	const std::vector<MiniDL::Data*> inputs{ &input };
	std::vector<MiniDL::Data*> bw_inputs{ &input };
	std::vector<MiniDL::Data*> outputs{ &output };

	std::cout << "\nafter init.\n-----------" << std::endl;
	MiniDL::log_data(input);
	MiniDL::log_data(output);

	//forward
	sigmoidOP.forward(inputs, outputs);
	std::cout << "\nafter forward.\n-----------" << std::endl;
	MiniDL::log_data(input);
	MiniDL::log_data(output);

	//backward
	sigmoidOP.backward(bw_inputs, outputs);

	std::cout << "\nafter backward.\n-----------" << std::endl;
	MiniDL::log_data(input);
	MiniDL::log_data(output);
}

static void show_data(const DataType& data)
{
	std::cout << "***********\n";
	for (size_t i = 0; i < data.size(); i++)
	{
		std::cout << data[i] << ",";
	}
	std::cout << "\n";
}

static void test_filler()
{
	DataType data(5);
	MiniDL::constant_filler(data, 1.0f);
	show_data(data);

	MiniDL::gaussian_filler(data, 1.0f, 1.0f);
	show_data(data);

	MiniDL::uniform_filler(data, 5.0f, 10.0f);
	show_data(data);
}

static void test_dense()
{
	const int batch_size = 1;
	const int input_dim = 3;
	const int output_dim = 2;

	MiniDL::Data input(MiniDL::Shape({ batch_size, input_dim }));
	input.fill(0.5f);
	MiniDL::Data output(MiniDL::Shape({ batch_size, output_dim }));
	output.fill(0.0f);
	const std::vector<MiniDL::Data*> inputs{ &input };	
	std::vector<MiniDL::Data*> outputs{ &output };

	std::cout << "\nafter init.\n-----------" << std::endl;
	MiniDL::log_data(input);
	MiniDL::log_data(output);

	MiniDL::DenseOP denseOP;
	denseOP.setup(input_dim, output_dim, false);
	denseOP.forward(inputs, outputs);

	std::cout << "\nweight init.\n-----------" << std::endl;
	MiniDL::log_data(denseOP.get_weight());
	MiniDL::log_data(denseOP.get_bias());

	std::cout << "\nafter forward.\n-----------" << std::endl;
	MiniDL::log_data(input);
	MiniDL::log_data(output);
}

int main(int argc, char* argv[])
{	
	test_dense();
	system("pause");
	return 0;
}