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
//
//#include "tensorflow/cc/ops/training_ops.h" // For ApplyGradientDescent
//#include "tensorflow/cc/ops/const_op.h" // For Const
//#include "tensorflow/cc/ops/math_ops.h" // For Cast
//#include "tensorflow/cc/framework/ops.h"
//#include "tensorflow/core/lib/core/status.h"
//
//TEST(test, Gradient)
//{
//	TensorflowWrapperInit;
//	// constructing variable und assigning have to be next to each other. No further
//	// calculatings in between
//	ops::Variable var = ops::Variable(root,{1}, DT_INT32);
//	Output varAssign = ops::Assign(root, var, { 1 });
//	// MatMul only works with matrix
//	Output Mul = ops::Mul(root, input, var);
//	Output learning_rate = ops::Const(root.WithOpName("learning_rate"), 0.01f, { 1 });
//	//Output Grad = ops::ApplyGradientDescent(root.WithOpName("GradDescent"), { var }, { learning_rate },{ Mul });
//	
//	ClientSession session(root);
//
//	Status st{ session.Run({ varAssign }, &output) };
//	ASSERT_TRUE(st.ok()) << st.error_message();
//	EXPECT_EQ(1, *output.at(0).scalar<int>().data());
//	output.clear();
//
//	st = session.Run({ { input,{2} } }, { Mul }, &output);
//	ASSERT_TRUE(st.ok()) << st.error_message();
//	EXPECT_EQ(2, *output.at(0).scalar<int>().data());
//	output.clear();
//
//	/*st = session.Run({ { input,{ 2 } } }, { Grad }, &output);
//	ASSERT_TRUE(st.ok()) << st.error_message();
//	EXPECT_EQ(2, *output.at(0).scalar<int>().data());
//	output.clear();*/
//}