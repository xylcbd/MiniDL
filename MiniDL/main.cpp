#include <iostream>
#include <cstdlib>
#include <string>
#include <random>
#include <ctime>
#include <cassert>
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

std::pair<std::vector<float>, std::vector<float>> generate_samples(const int total_samples, const int input_len, const int output_len)
{
	assert(input_len == output_len);
	std::default_random_engine generator(std::random_device{}());
	std::uniform_real_distribution<float> distribution(-100.0f, 100.0f);

	std::vector<float> inputs(total_samples * input_len);
	std::vector<float> outputs(total_samples * output_len);

	//fill inputs
	for (size_t i = 0; i < inputs.size(); i++)
	{
		inputs[i] = distribution(generator);
		// y = x*x + 1.0
		outputs[i] = inputs[i] * inputs[i] + 1.0f;
	}
	return std::make_pair(inputs, outputs);
}

std::vector<std::pair<std::vector<float>, std::vector<float>>> as_batches(const std::pair<std::vector<float>, std::vector<float>>& samples, 
	const int batch_size, const int input_len, const int output_len)
{
	const std::vector<float>& inputs = samples.first;
	const std::vector<float>& outputs = samples.second;
	assert(batch_size > 0);
	assert(inputs.size() == outputs.size());
	//input as batch
	std::vector<std::vector<float>> batch_inputs;
	for (size_t i = 0; i < inputs.size(); i += batch_size)
	{
		if (i + batch_size > inputs.size())
		{
			break;
		}
		std::vector<float> batch_input(batch_size * input_len);
		std::copy(inputs.begin() + i, inputs.begin() + i + batch_size, batch_input.begin());
		batch_inputs.push_back(batch_input);
	}
	//output as batch
	std::vector<std::vector<float>> batch_outputs;
	for (size_t i = 0; i < outputs.size(); i += batch_size)
	{
		if (i + batch_size > outputs.size())
		{
			break;
		}
		std::vector<float> batch_output(batch_size * output_len);
		std::copy(outputs.begin() + i, outputs.begin() + i + batch_size, batch_output.begin());
		batch_outputs.push_back(batch_output);
	}
	//package
	assert(batch_inputs.size() == batch_outputs.size());
	std::vector<std::pair<std::vector<float>, std::vector<float>>> total_batches;
	for (size_t i = 0; i < batch_inputs.size(); i++)
	{
		std::pair<std::vector<float>, std::vector<float>> batch = std::make_pair(batch_inputs[i], batch_outputs[i]);
		total_batches.push_back(batch);
	}
	return total_batches;
}

static void fill_batch(MiniDL::Data& data, const std::vector<float> vals)
{
	assert(data.get_shape().get_total() == vals.size());
	data.get_data() = vals;
}

static void test_network()
{
	////////////////////////////////////////////////////////////////
	// data defines
	const int batch_size = 2;
	const int input_len = 2;
	const int hidden_len = 3;
	const int output_len = 2;

	const std::pair<std::vector<float>, std::vector<float>> samples = generate_samples(1000, input_len, output_len);
	const std::vector<std::pair<std::vector<float>, std::vector<float>>> batches = as_batches(samples, batch_size, input_len, output_len);

	//input wrapper
	std::vector<MiniDL::Data*> input_datas;
	MiniDL::Data input_data(MiniDL::Shape{batch_size, input_len});
	input_datas.push_back(&input_data);

	//predicts wrapper
	std::vector<MiniDL::Data*> predict_datas;
	MiniDL::Data predict_data(MiniDL::Shape{ batch_size, output_len });
	predict_datas.push_back(&predict_data);

	//output wrapper
	std::vector<MiniDL::Data*> groundtruth_datas;
	MiniDL::Data groundtruth_data(MiniDL::Shape{ batch_size, output_len });
	groundtruth_datas.push_back(&groundtruth_data);

	////////////////////////////////////////////////////////////////
	//network defines
	MiniDL::Network net;

	//input
	MiniDL::InputOP input_op;
	net.add_op(&input_op);

	//hidden
	MiniDL::DenseOP hidden_op;
	hidden_op.setup(input_len, hidden_len, true);
	net.add_op(&hidden_op);
	MiniDL::SigmoidOP hidden_active_op;
	net.add_op(&hidden_active_op);

	//output
	MiniDL::DenseOP output_op;
	hidden_op.setup(hidden_len, output_len, true);
	net.add_op(&output_op);
	MiniDL::SigmoidOP output_active_op;
	net.add_op(&output_active_op);

	//loss
	MiniDL::MSELoss loss_op;
	net.set_loss_functor(&loss_op);

	////////////////////////////////////////////////////////////////
	// train network
	const int epoch = 2;
	for (int i = 0; i < epoch; i++)
	{
		for (size_t j = 0; j < batches.size(); j++)
		{
			//get batch
			std::pair<std::vector<float>, std::vector<float>> batch = batches[j];
			fill_batch(input_data, batch.first);
			fill_batch(groundtruth_data, batch.second);

			//forward
			net.forward(input_datas, predict_datas);
			//backward
			net.backward(input_datas, groundtruth_datas);
			//update weights
			net.update_weights();

			//get loss
			const float loss = net.get_loss(groundtruth_datas, predict_datas);
			std::cout << "Epoch[" << i <<"/" << epoch <<  "] " <<
				"Batch[" << j << "/" << batches.size() << "] " << 
				"loss: " << loss << std::endl;
		}
	}
	const std::string model_path = "minidl.mlp.model";
	net.save_model(model_path);
}

int main(int argc, char* argv[])
{	
	test_network();
	system("pause");
	return 0;
}