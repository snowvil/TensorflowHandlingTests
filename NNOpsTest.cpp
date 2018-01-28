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