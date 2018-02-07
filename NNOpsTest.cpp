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

	Status st{ session.Run({{input,{-1}}}, { AbsValue }, &output) };
	EXPECT_EQ(1, *output.at(0).scalar<int>().data());
	output.clear();
}

TEST(Math, AbsVector)
{
	TensorflowWrapperInit;
	Output AbsValue = ops::Abs(root, input);

	Status st{ session.Run({ { input,{ -1,-2 } } },{ AbsValue }, &output) };
	int* val = output.at(0).vec<int>().data();
	EXPECT_EQ(1, *val);
	EXPECT_EQ(2, *(++val));
	output.clear();
}

TEST(Math, SumVec)
{
	TensorflowWrapperInit;

	Output Sum = ops::Sum(root, input,0);

	Status st{ session.Run({ { input,{ -1,-2 } } },{ Sum }, &output) };
	ASSERT_TRUE(st.ok()) << st.error_message();
	int* val = output.at(0).flat<int>().data();
	EXPECT_EQ(*val, -3);
}

TEST(Math, SumMatrix)
{
	TensorflowWrapperInit;
	//Eigen::Matrix2i mat;
	//mat << 1, 2, 3, 4;
	Tensor values(DT_INT32, { 2,2 });
	for(int64 i = 0; i < values.NumElements(); i++)
		values.flat<int>()(i) = i;

	Output Sum0 = ops::Sum(root, input, 0);
	Output Sum1 = ops::Sum(root, input, 1);

	Status st{ session.Run({ { input,values}},{ Sum0,Sum1 }, &output) };
	ASSERT_TRUE(st.ok()) << st.error_message();

	int* val = output.at(0).flat<int>().data();
	EXPECT_EQ(2, *val++);
	EXPECT_EQ(4, *val++);

	val = output.at(1).flat<int>().data();
	EXPECT_EQ(1, *val++);
	EXPECT_EQ(5, *val++);
}

TEST(NNOps, ReluSingle)
{
	TensorflowWrapperInit;
	Output Relu = ops::Relu(root, input);

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

TEST(NNOps, L2LossScalar)
{
	//L2Loss = sum(t**2)/2

	Scope root = Scope::NewRootScope();
	::std::vector<Tensor> output;

	PartialTensorShape inputShape({ 1,1 });
	ops::Placeholder::Attrs attrs;
	attrs.Shape(inputShape);
	ops::Placeholder x = ops::Placeholder(root, DT_FLOAT,attrs);

	// L2Loss needs an 2D-Array. Therefore must have the placeholder-value a shape-attribut
	Output L2Loss = ops::L2Loss(root, x);

	ClientSession session(root);

	// It seems like you can add more values to the input parameter than stated in the placeholder attribute
	Status status = session.Run({ {x,{1.0F}} }, { L2Loss }, &output);
	ASSERT_TRUE(status.ok()) << status.error_message();
	float* val = output.at(0).scalar<float>().data();
	EXPECT_EQ(0.5, *val);
}

TEST(NNOps, L2LossVector)
{
	//L2Loss = sum(t**2)/2

	Scope root = Scope::NewRootScope();
	::std::vector<Tensor> output;

	PartialTensorShape inputShape({ 3,1 });
	ops::Placeholder::Attrs attrs;
	attrs.Shape(inputShape);
	ops::Placeholder x = ops::Placeholder(root, DT_FLOAT, attrs);

	// L2Loss needs an 2D-Array. Therefore must have the placeholder-value a shape-attribut
	Output L2Loss = ops::L2Loss(root, x);

	ClientSession session(root);

	// It seems like you can add more values to the input parameter than stated in the placeholder attribute.
	// L2Loss also needs double or float values
	Status status = session.Run({ { x,{ 1.0F,2.0F,3.0F } } }, { L2Loss }, &output);
	ASSERT_TRUE(status.ok()) << status.error_message();
	float* val = output.at(0).scalar<float>().data();
	EXPECT_FLOAT_EQ(7, *val);
}