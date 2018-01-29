#include "gtest/gtest.h"
#include "testhelper.h"

#include <vector>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"	// for placeholder
#include "tensorflow/cc/ops/math_ops.h"		// for Add, Abs, etc
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/client/client_session.h"

using namespace ::tensorflow;

TEST(Math, AbsSingle)
{
	TensorflowWrapperInit;
	Output AbsValue = ops::Abs(root, input);

	ClientSession session(root);
	Status st{ session.Run({{input,{-1}}}, { AbsValue }, &output) };
	EXPECT_EQ(1, *output.at(0).scalar<int>().data());
	output.clear();
}

TEST(Math, AbsVector)
{
	TensorflowWrapperInit;
	Output AbsValue = ops::Abs(root, input);

	ClientSession session(root);
	Status st{ session.Run({ { input,{ -1,-2 } } },{ AbsValue }, &output) };
	int* val = output.at(0).vec<int>().data();
	EXPECT_EQ(1, *val);
	EXPECT_EQ(2, *(++val));
	output.clear();
}

TEST(NNOps, ReluSingle)
{
	TensorflowWrapperInit;
	Output Relu = ops::Relu(root, input);

	ClientSession session(root);
	Status st{ session.Run({ { input,{ 0 } } },{ Relu }, &output) };
	int* val = output.at(0).vec<int>().data();
	EXPECT_EQ(0, *val);
	output.clear();

	st = session.Run({ { input,{ 1 } } },{ Relu }, &output);
	val = output.at(0).vec<int>().data();
	EXPECT_EQ(1, *val);
	output.clear();

	st =  session.Run({ { input,{ -1 } } },{ Relu }, &output);
	val = output.at(0).vec<int>().data();
	EXPECT_EQ(0, *val);
	output.clear();

	st = session.Run({ { input,{ 10 } } },{ Relu }, &output);
	val = output.at(0).vec<int>().data();
	EXPECT_EQ(10, *val);
	output.clear();
}

TEST(NNOps, ReluVector)
{
	TensorflowWrapperInit;
	Output Relu = ops::Relu(root, input);

	ClientSession session(root);
	Status st{ session.Run({ { input,{ -1,0,1,10 } } },{ Relu }, &output) };
	int* val = output.at(0).vec<int>().data();
	EXPECT_EQ(0, *val);
	EXPECT_EQ(0, *(++val));
	EXPECT_EQ(1, *(++val));
	EXPECT_EQ(10, *(++val));
	output.clear();
}

TEST(Math, TanhSingle)
{
	TensorflowWrapperInit;
	Output Tanh = ops::Tanh(root, ops::Cast(root,input,DT_FLOAT));

	ClientSession session(root);
	Status st{ session.Run({ { input,{ 0 } } },{ Tanh }, &output) };
	float* val = output.at(0).vec<float>().data();
	EXPECT_EQ(0, *val);
	output.clear();

	st = session.Run({ { input,{ 100 } } }, { Tanh }, &output);
	val = output.at(0).vec<float>().data();
	EXPECT_EQ(1, *val);
	output.clear();

	st = session.Run({ { input,{ -100 } } }, { Tanh }, &output);
	val = output.at(0).vec<float>().data();
	EXPECT_EQ(-1, *val);
	output.clear();
}