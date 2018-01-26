#include "gtest/gtest.h"
#include "testhelper.h"

#include <vector>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/state_ops.h" //For Variable

using namespace ::tensorflow;

TEST(Variable, Variable)
{
	TensorflowWrapperInit;
	ops::Variable var = ops::Variable(root, { 1 }, DT_INT32);
	Output varAssign = ops::Assign(root, var, { 1 });

	ClientSession session(root);
	Status st{ session.Run({ varAssign }, &output) };
	EXPECT_EQ(1, *output.at(0).scalar<int>().data());
	output.clear();

	varAssign = ops::Assign(root, var, { 2 });
	st = session.Run({ varAssign }, &output);
	EXPECT_EQ(2, *output.at(0).scalar<int>().data());
}

TEST(Variable, VariableAdd)
{
	TensorflowWrapperInit;
	ops::Variable var = ops::Variable(root, { 1 }, DT_INT32);
	Output varAssign = ops::Assign(root, var, { 1 });

	ClientSession session(root);
	Status st{ session.Run({ varAssign }, &output) };
	EXPECT_EQ(1, *output.at(0).scalar<int>().data());
	output.clear();

	Output add = ops::Add(root, var, input);
	st = session.Run({ { { input,{ 2 } } } }, { add }, &output);
	EXPECT_EQ(3, *output.at(0).scalar<int>().data());
}

