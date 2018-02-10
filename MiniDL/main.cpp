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

static void test_sigmoid()
{
	MiniDL::SigmoidOP sigmoidOP;
	sigmoidOP.set_phase(MiniDL::Operator::Phase::Train);
	MiniDL::Shape shape({ 2,2 });
	MiniDL::Data input(shape);
	input.fill(0.5f);
	MiniDL::Data prev_diff(shape);
	prev_diff.fill(0.0f);
	MiniDL::Data output(shape);
	output.fill(0.0f);
	MiniDL::Data next_diff(shape);
	next_diff.fill(1.0f);
	const std::vector<MiniDL::Data*> inputs{ &input };
	std::vector<MiniDL::Data*> prev_diffs{ &prev_diff };
	std::vector<MiniDL::Data*> next_diffs{ &next_diff };
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
	sigmoidOP.backward(inputs, prev_diffs, next_diffs, outputs);

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
	MiniDL::Data prev_diff(input.get_shape());
	prev_diff.fill(0.0f);
	MiniDL::Data output(MiniDL::Shape({ batch_size, output_dim }));
	output.fill(0.0f);
	MiniDL::Data next_diff(output.get_shape());
	next_diff.fill(1.0f);

	const std::vector<MiniDL::Data*> inputs{ &input };	
	std::vector<MiniDL::Data*> prev_diffs{ &prev_diff };	
	std::vector<MiniDL::Data*> outputs{ &output };
	std::vector<MiniDL::Data*> next_diffs{ &next_diff };

	std::cout << "\nafter init.\n-----------" << std::endl;
	MiniDL::log_data(input);
	MiniDL::log_data(output);

	MiniDL::DenseOP denseOP;
	denseOP.set_phase(MiniDL::Operator::Phase::Train);
	denseOP.setup(input_dim, output_dim, true);
	denseOP.forward(inputs, outputs);

	std::cout << "\nweight init.\n-----------" << std::endl;
	MiniDL::log_data(denseOP.get_weight());
	MiniDL::log_data(denseOP.get_bias());

	std::cout << "\nafter forward.\n-----------" << std::endl;
	MiniDL::log_data(input);
	MiniDL::log_data(output);

	denseOP.backward(inputs, prev_diffs, next_diffs, outputs);
	std::cout << "\nafter backward.\n-----------" << std::endl;
	MiniDL::log_data(prev_diff);

	std::cout << "all weights:" << std::endl;
	std::vector<MiniDL::Data*> all_weights = denseOP.get_all_weights();
	for (size_t i = 0; i < all_weights.size(); i++)
	{
		MiniDL::log_data(*all_weights[i]);
	}

	std::cout << "all grads:" << std::endl;
	std::vector<MiniDL::Data*> all_grads = denseOP.get_all_grads();
	for (size_t i = 0; i < all_grads.size(); i++)
	{
		MiniDL::log_data(*all_grads[i]);
	}
}

int main(int argc, char* argv[])
{	
	test_dense();
	system("pause");
	return 0;
}