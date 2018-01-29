#include "gtest/gtest.h"
#include "testhelper.h"

#include <vector>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/client/client_session.h"

using namespace ::tensorflow;

TEST(Const, Scalar)
{
	TensorflowWrapperInit;
	Output Add = ops::Add(root.WithOpName("add"), input, { 0 });
	ClientSession session(root);

	session.Run({ { input,{ 1 } } }, { Add }, &output);
	EXPECT_EQ(1, *(output.at(0).scalar<int>().data()));
}

TEST(Const, Vector)
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

#include "tensorflow/cc/ops/const_op.h"
TEST(Const, FloatConstruct)
{
	Scope root = Scope::NewRootScope();
	::std::vector<Tensor> output;

	Output Const = ops::Const(root, { 1.0F });

	ClientSession session(root);
	Status st({ session.Run({},{ Const }, &output) });
	EXPECT_EQ(1.0F, *(output.at(0).vec<float>().data())) << st.error_message();
}

TEST(Const, MatrixConstruct)
{
	Scope root = Scope::NewRootScope();
	::std::vector<Tensor> output;

	Output Const = ops::Const(root, { {1,2},{3,4} });

	ClientSession session(root);
	Status st({ session.Run({},{ Const }, &output) });
	
	// Test the matrix and access some of the values
	TensorShape tShape = output.at(0).shape();
	EXPECT_TRUE(tShape.IsSameSize({ 2,2 }));

	EXPECT_EQ(output.at(0).NumElements(), 4);
	EXPECT_EQ(output.at(0).dtype(), DT_INT32);

	// this returns an EIGEN-matrix.
	auto test = output.at(0).matrix<int>();
	EXPECT_EQ(test(0, 0), 1);
	EXPECT_EQ(test(0, 1), 2);
	EXPECT_EQ(test(1, 0), 3);
	EXPECT_EQ(test(1, 1), 4);
}

TEST(Const, Float)
{
	Scope root = Scope::NewRootScope();
	ops::Placeholder input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
	::std::vector<Tensor> output;

	ClientSession session(root);
	Output Add = ops::Add(root.WithOpName("add"), input, input);
	Status st({ session.Run({ { input,{ 1.2F } } },{ Add }, &output) });
	EXPECT_EQ(2.4F, *(output.at(0).vec<float>().data())) << st.error_message();
}