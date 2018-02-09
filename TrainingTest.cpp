#include "gtest/gtest.h"
#include "testhelper.h"

#include <vector>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h" //For Placeholder
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/state_ops.h" //For Variable

#include "tensorflow/cc/ops/training_ops.h" // For ApplyGradientsDecent

using namespace ::tensorflow;

TEST(Training, ApplyGradientsDescent)
{
	TensorflowWrapperInit;
	Output variable = ops::Variable(root, { 1 }, DT_FLOAT);
	Output assign = ops::Assign(root, variable, { 1.0F });

	Status status({ session.Run({assign},nullptr) });

	// if the documentation is talking about scalar, it seems, that the function does not need 
	// an initializer list for that value
	Output applyGradDescent = ops::ApplyGradientDescent(root, variable, 0.1F, { 1.0F });
	status = session.Run({}, { applyGradDescent }, &output);
	EXPECT_TRUE(status.ok()) << status.error_message();
	float* val = output.at(0).scalar<float>().data();
	EXPECT_FLOAT_EQ(0.9, *val);
}


#include <vector>
#include <numeric>
#include <algorithm>

#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/cc/framework/gradients.h"

constexpr float RISE = 2.75;
constexpr float OFFSET = -1.45;

float generateValues(int value)
{
	return (value * RISE + OFFSET);
}


TEST(Training, AddSymbolicGradients)
{
	constexpr int TRAININGSIZE	= 1000;
	constexpr int TESTSIZE		= 50;

	// create trainvector
	::std::vector<int> trainX(TRAININGSIZE,0);
	::std::iota(trainX.begin(), trainX.end(), 0);
	::std::vector<float> trainY(TRAININGSIZE,0.0);
	::std::transform(trainX.begin(), trainX.end(),trainY.begin(),generateValues);

	// create testvector
	::std::vector<int> testX(TESTSIZE, 0);
	::std::for_each(testX.begin(), testX.end(), [](int& val){ val = (std::rand() % 1000); });
	::std::vector<float> testY(TESTSIZE, 0.0);
	::std::transform(testX.begin(), testX.end(), testY.begin(), generateValues);

	// convert the vectors into tensors
	Tensor tensorTestX{ DT_INT32,TensorShape{TRAININGSIZE} };
	Tensor tensorTestY{ DT_FLOAT,TensorShape{ TRAININGSIZE } };

	memcpy(tensorTestX.vec<int>().data(), trainX.data(), TRAININGSIZE * sizeof(int));
	memcpy(tensorTestY.vec<float>().data(), trainY.data(), TRAININGSIZE * sizeof(float));

	Scope root = Scope::NewRootScope();
	ops::Placeholder inputHolder = ops::Placeholder(root.WithOpName("input"), DT_INT32);
	ops::Placeholder outputHolder = ops::Placeholder(root.WithOpName("output"), DT_FLOAT);

	::std::vector<Tensor> output;
	ClientSession session(root);

	Output w = ops::Variable(root, { 1 }, DT_FLOAT);
	Output w_assign = ops::Assign(root, w, { 1.0F });
	Output b = ops::Variable(root, { 1 }, DT_FLOAT);
	Output b_assign = ops::Assign(root, b, { 1.0F });
	Status status({ session.Run({ w_assign,b_assign },nullptr) });
	ASSERT_TRUE(status.ok()) << status.error_message();

	// y = m*x + b
	Output riser = ops::Mul(root.WithOpName("riser"),ops::Cast(root,inputHolder,DT_FLOAT), w);
	Output y = ops::Add(root.WithOpName("bias"),riser,b);

	//auto loosness = ops::L2Loss(root, w);
	auto temp1 = ops::Sub(root, y, outputHolder);
	auto temp2 = ops::Square(root, temp1);
	auto RMS = ops::Mean(root, temp2, {0});
	//auto temp4 = ops::Mul(root, ops::Cast(root, 0.01, DT_FLOAT), loosness);
	//auto loss = ops::Add(root, RMS,temp4);

	std::vector<Output> grad_outputs;
	AddSymbolicGradients(root, { RMS }, { w,b}, &grad_outputs);
	// the learning rate is way more important than i expected. Feel free and try some values. It will crash easily.
	// If the RISE and OFFSET values rise, these learning rates have to be adjusted
	auto apply_w = ops::ApplyGradientDescent(root, w, ops::Cast(root, 0.000001, DT_FLOAT), { grad_outputs[0] });
	auto apply_b = ops::ApplyGradientDescent(root, b, ops::Cast(root, 0.01, DT_FLOAT), { grad_outputs[1] });


	ClientSession::FeedType inputs;
	inputs.emplace(inputHolder, tensorTestX);
	inputs.emplace(outputHolder, tensorTestY);
	float RMSValue = 1;
	size_t idx = 0;
	while(RMSValue > 1E-7 && idx < 100000)
	{
		status = session.Run(inputs, { y,RMS,apply_w, apply_b, temp1 }, &output);
		ASSERT_TRUE(status.ok()) << status.error_message();
		if (idx % 1000 == 0 || idx < 10)
		{
			std::cout << idx << "\t";
			std::cout <<"y: " << output.at(0).vec<float>()(1) << "\t";
			std::cout <<"y_des: " << tensorTestY.vec<float>()(1) << "\t";
			std::cout << "RMS: " << output.at(1).scalar<float>() << "\t";
			std::cout << "apply_w: " << output.at(2).scalar<float>() << "\t";
			std::cout << "apply_b: " << output.at(3).scalar<float>();
			std::cout << std::endl;
		}
		RMSValue = *output.at(1).scalar<float>().data();
		idx++;
	}
	std::cout << "it took " << idx << " iterations" << std::endl;
}