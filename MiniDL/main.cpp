#include <iostream>
#include <cstdlib>
#include <string>
#include <opencv2/opencv.hpp>
#include "MiniDL.h"

static void test_tools()
{
	MiniDL::do_assert(true, "you can't see this message.");
	MiniDL::do_assert(false, "test only.");
}

static void test_ops()
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

int main(int argc, char* argv[])
{
	//DO SOMETHING
	//test_tools();
	test_ops();

	system("pause");
	return 0;
}