#include "gtest/gtest.h"

#include <vector>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/client/client_session.h"

using namespace ::tensorflow;

#define TensorflowWrapperInit \
	Scope root = Scope::NewRootScope(); \
	ops::Placeholder input = ops::Placeholder(root.WithOpName("input"), DT_INT32); \
	::std::vector<Tensor> output;

TEST(test, Scalar)
{
	TensorflowWrapperInit;
	Output Add = ops::Add(root.WithOpName("add"), input, { 0 });
	ClientSession session(root);

	session.Run({ { input,{ 1 } } }, { Add }, &output);
	EXPECT_EQ(1, *(output.at(0).scalar<int>().data()));
}

TEST(test, Vector)
{
	TensorflowWrapperInit;
	Output Add = ops::Add(root.WithOpName("add"), input, { 1,1 });
	ClientSession session(root);

	session.Run({ { input,{1,2} } }, { Add }, &output);
	int* val = output.at(0).vec<int>().data();
	EXPECT_EQ(2, output.at(0).vec<int>().size());
	EXPECT_EQ(2, *val);
	EXPECT_EQ(3, *(++val));
}

#include "tensorflow/cc/ops/state_ops.h" //For Variable
#include "tensorflow/cc/ops/training_ops.h" // For ApplyGradientDescent
#include "tensorflow/cc/framework/ops.h"

TEST(test, Gradient)
{
	TensorflowWrapperInit;
	// constructing variable und assigning have to be next to each other. No further
	// calculatings in between
	ops::Variable var = ops::Variable(root,{1},DT_INT32);
	Output varAssign = ops::Assign(root, var, { 1 });
	// MatMul only works with matrix
	Output Mul = ops::Mul(root, input, var);
	Output Grad = ops::ApplyGradientDescent(root, var, { 0.1 }, { 1 });
	
	ClientSession session(root);

	session.Run({ varAssign }, &output);
	EXPECT_EQ(1, *output.at(0).scalar<int>().data());
	output.clear();

	session.Run({ { input,{2} } }, { Mul }, &output);
	EXPECT_EQ(2, *output.at(0).scalar<int>().data());
	output.clear();

	/*session.Run({ { input,{ 2 } } }, { Grad }, &output);
	EXPECT_EQ(2, *output.at(0).scalar<int>().data());
	output.clear();*/
}