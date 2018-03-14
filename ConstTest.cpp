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

	session.Run({ { input,{ 1 } } }, { Add }, &output);
	EXPECT_EQ(1, *(output.at(0).scalar<int>().data()));
}

TEST(Const, RichByPlaceholder)
{
	TensorflowWrapperInit;

	Status st = session.Run({ { input,{ 1 } } }, { input }, &output);
	ASSERT_TRUE(st.ok()) << st.error_message();
	EXPECT_EQ(1, *(output.at(0).scalar<int>().data()));
}

TEST(Const, Vector)
{
	TensorflowWrapperInit;
	Output Add = ops::Add(root.WithOpName("add"), input, { 1,1 });

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

	Output Const = ops::Const(root, { {1,2}, {3,4} });

	ClientSession session(root);
	Status st({ session.Run({},{ Const }, &output) });
	
	// Test the matrix and access some of the values
	TensorShape tShape = output.at(0).shape();
	EXPECT_TRUE(tShape.IsSameSize({ 2,2 }));

	EXPECT_EQ(output.at(0).NumElements(), 4);
	EXPECT_EQ(output.at(0).dtype(), DT_INT32);

	// this returns an EIGEN-matrix
	TTypes<int>::Matrix outputMatrix = output.at(0).matrix<int>();
	EXPECT_EQ(outputMatrix(0, 0), 1);
	EXPECT_EQ(outputMatrix(0, 1), 2);
	EXPECT_EQ(outputMatrix(1, 0), 3);
	EXPECT_EQ(outputMatrix(1, 1), 4);

	EXPECT_EQ(outputMatrix.size(),4);
	int* data = outputMatrix.data();

	EXPECT_EQ(*data++, 1);
	EXPECT_EQ(*data++, 2);
	EXPECT_EQ(*data++, 3);
	EXPECT_EQ(*data++, 4);
}

TEST(Const, PlaceholderWithDifferentShapesByRun)
{
	Scope root = Scope::NewRootScope();
	ops::Placeholder input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
	::std::vector<Tensor> output;

	ClientSession session(root);
	Output Add = ops::Add(root.WithOpName("add"), input, input);
	Status st({ session.Run({ { input,{ 1.2F } } },{ Add }, &output) });
	EXPECT_EQ(2.4F, *(output.at(0).scalar<float>().data())) << st.error_message();

	st = session.Run({ { input,{ 1.1F,1.0F } } },{ Add }, &output);
	float* data = output.at(0).vec<float>().data();
	ASSERT_TRUE(st.ok()) << st.error_message();
	EXPECT_EQ(2.2F, *data++);
	EXPECT_EQ(2.0F, *data);

	st = session.Run({ {input,{{ 1.1F,1.0F },{2.0F,3.0F}}} }, { Add }, &output);
	data = output.at(0).matrix<float>().data();
	ASSERT_TRUE(st.ok()) << st.error_message();
	EXPECT_EQ(2.2F, *data++);
	EXPECT_EQ(2.0F, *data++);
	EXPECT_EQ(4.0F, *data++);
	EXPECT_EQ(6.0F, *data++);
}

TEST(Const, CreateTensor)
{
    Scope root = Scope::NewRootScope();
	ops::Placeholder input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
	::std::vector<Tensor> output;
	ClientSession session(root);
	Output Add = ops::Add(root.WithOpName("add"), input, input);

    Tensor myTensor(DT_FLOAT, { 1,1 });
	myTensor.scalar<float>()(0) = 3.0F;

	Status st = session.Run({ { input,myTensor} }, { Add }, &output);
	ASSERT_TRUE(st.ok()) << st.error_message();
	float* data = output.at(0).matrix<float>().data();
	EXPECT_EQ(6.0F, *data);
}

TEST(Const, CopyVector)
{
	TensorflowWrapperInit;
	Output Add = ops::Add(root.WithOpName("add"), input, input);

	::std::vector<int> vec({ 1,2,3,4 });
	Tensor myEigenTensor(DT_INT32, TensorShape({ 4 }));
	::std::copy_n(vec.begin(), vec.size(), myEigenTensor.vec<int>().data());
	std::cout << myEigenTensor.DebugString();
	Status st = session.Run({ { input,myEigenTensor } }, { Add }, &output);
	int* data = output.at(0).vec<int>().data();
	for (const int& it: vec)
	{
		EXPECT_EQ(2 * it, *data);
		data++;
	}
}

TEST(Const, CopyEigenMatrix)
{
	TensorflowWrapperInit;

	Output Add = ops::Add(root.WithOpName("add"), input, input);

	Eigen::Matrix2i mtrx = Eigen::Matrix2i::Random();
	::std::cout << mtrx << ::std::endl;
	Tensor myEigenTensor(DT_INT32, { mtrx.rows(),mtrx.cols() });
	::std::copy_n(mtrx.data(), mtrx.size(), myEigenTensor.matrix<int>().data());
	std::cout << myEigenTensor.DebugString();
	Status st = session.Run({ { input,myEigenTensor } }, { Add }, &output);
	int* data = output.at(0).matrix<int>().data();
	for (Eigen::Index ctr = 0; ctr < mtrx.size(); ctr++)
	{
		EXPECT_EQ(2 * mtrx(ctr), *data);
		data++;
	}
}

TEST(Const, Ones)
{
	constexpr int nSize = 4;
	Scope root{ Scope::NewRootScope() };

	Eigen::MatrixXi ones = Eigen::MatrixXi::Ones(nSize, nSize);
	::std::cout << ones << ::std::endl;
	Tensor onesTensor{ DT_INT32,{ones.rows(),ones.cols()} };
	std::copy_n(ones.data(), ones.size(), onesTensor.matrix<int>().data());

	Output Ones = ops::Const(root, onesTensor);

	ClientSession sess = root;
	std::vector<Tensor> output;
	Status st = sess.Run({ Ones }, &output);
	CHECK_STATUS(st);

	EXPECT_EQ(output.size(), 1);
	EXPECT_EQ(output.at(0).dim_size(0), output.at(0).dim_size(1));
	EXPECT_EQ(output.at(0).dim_size(0), nSize);
	auto data = output.at(0).matrix<int>().data();
	for (Eigen::Index ctr = 0; ctr < ones.size(); ctr++)
	{
		EXPECT_EQ(*data,1);
		data++;
	}
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