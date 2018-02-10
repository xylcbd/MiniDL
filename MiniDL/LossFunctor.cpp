#include "LossFunctor.h"
#include "Tools.h"

namespace MiniDL
{
	//MSELoss
	// y = 1/N * sum((pd[i]-va[i]) * (pd[i]-va[i]))  -- i=[0-N)
	float MSELoss::get_loss(const std::vector<Data*>& groundtruthes, const std::vector<Data*>& predicts)
	{
		do_assert(groundtruthes.size() == predicts.size(), "size of groundtruthes must be equals with predicts.");

		int total = 0;
		float losses = 0.0f;
		for (size_t i = 0; i < groundtruthes.size(); i++)
		{
			do_assert(groundtruthes[i] && predicts[i], "groundtruthes or predicts can't be NULL.");
			const Data& groundtruth = *groundtruthes[i];
			const Data& predict = *predicts[i];
			do_assert(groundtruth.get_shape() == predict.get_shape(), "shape of groundtruthes must be equals with predicts.");

			const DataType& groundtruth_mems = groundtruth.get_data();
			const DataType& predict_mems = predict.get_data();
			for (size_t j = 0; j < groundtruth.get_shape().get_total(); j++)
			{
				total++;
				const float gt_val = groundtruth_mems[j];
				const float pd_val = predict_mems[j];
				const float loss_val = (pd_val - gt_val) * (pd_val - gt_val);
				losses += loss_val;
			}
		}
		do_assert(total > 0, "total must be larger than zero.");
		const float outputs = losses / total;
		return outputs;
	}

	// y = 1/N * sum((pd[i]-va[i]) * (pd[i]-va[i]))  -- i=[0-N)
	std::vector<Data*> MSELoss::get_grads(const std::vector<Data*>& groundtruthes, const std::vector<Data*>& predicts)
	{
		do_assert(groundtruthes.size() == predicts.size(), "size of groundtruthes must be equals with predicts.");
		std::vector<Data*> outputs;		
		//y = (pd-gt)^2
		//grad = dy/dpd = 2(pd-gt)*1.0f
		for (size_t i = 0; i < groundtruthes.size(); i++)
		{
			do_assert(groundtruthes[i] && predicts[i], "groundtruthes or predicts can't be NULL.");
			const Data& groundtruth = *groundtruthes[i];
			const Data& predict = *predicts[i];
			do_assert(groundtruth.get_shape() == predict.get_shape(), "shape of groundtruthes must be equals with predicts.");

			const DataType& groundtruth_mems = groundtruth.get_data();
			const DataType& predict_mems = predict.get_data();

			Data* output = new Data(predict.get_shape());
			DataType& output_mems = output->get_data();
			for (size_t j = 0; j < predict.get_shape().get_total(); j++)
			{
				output_mems[j] = 2.0f * (predict_mems[j] - groundtruth_mems[j]);
			}
			outputs.push_back(output);
		}
		return outputs;
	}

} //namespace MiniDL