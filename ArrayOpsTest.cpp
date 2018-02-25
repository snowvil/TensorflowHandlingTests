#include "gtest/gtest.h"
#include "testhelper.h"

#include <vector>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/client/client_session.h"

using namespace ::tensorflow;

TEST(ArrayOps, ConCat)
{
	TensorflowWrapperInit;

	ops::Placeholder input1 = ops::Placeholder(root, DT_INT32);

	InputList oList{ std::initializer_list<Input>{input,input1}};
	Output ConCat0 = ops::Concat(root, oList, 0);
	//Output ConCat1 = ops::Concat(root, oList, 1);

	//Status st = session.Run({ {input,{1,2}},{input1,{3,4}} }, { ConCat0,ConCat1 }, &output);
	Status st = session.Run({ { input,{ 1,2 } },{ input1,{ 3,4 } } }, { ConCat0 }, &output);
	ASSERT_TRUE(st.ok()) << st.error_message();

	std::cout << output.at(0).DebugString() << std::endl;
	std::cout << output.at(0).dims() << std::endl;
	//std::cout << output.at(1).DebugString();
}

TEST(ArrayOps, ExpandDims)
{
	TensorflowWrapperInit;

	Output ExpandDims = ops::ExpandDims(root, input, 1);
	Status st = session.Run({ { input,{ 1,2 }} }, { ExpandDims }, &output);
	ASSERT_TRUE(st.ok()) << st.error_message();

	EXPECT_EQ(2, output.at(0).dims());
	EXPECT_EQ(2, output.at(0).dim_size(0));
	EXPECT_EQ(1, output.at(0).dim_size(1));
}

TEST(ArrayOps, ExpandDimsAndConcat)
{
	TensorflowWrapperInit;

	ops::Placeholder input1 = ops::Placeholder(root, DT_INT32);

	Output ExpandDimsInput = ops::ExpandDims(root, input, 1);
	Output ExpandDimsInput1 = ops::ExpandDims(root, input1, 1);

	InputList oList{ std::initializer_list<Input>{input,input1} };
	Output ConCat0 = ops::Concat(root, oList, 0);
	Output ConCat1 = ops::Concat(root, { std::initializer_list<Input>{ExpandDimsInput,ExpandDimsInput1} }, 1);

	Status st = session.Run({ {input,{1,2}},{input1,{3,4}} }, { ConCat0,ConCat1 }, &output);
	//Status st = session.Run({ { input,{ 1,2 } },{ input1,{ 3,4 } } }, { ConCat0 }, &output);
	ASSERT_TRUE(st.ok()) << st.error_message();

	std::cout << output.at(0).DebugString() << std::endl;
	int dimensions = output.at(0).dims();
	EXPECT_EQ(1, dimensions);
	EXPECT_EQ(4, output.at(0).dim_size(0));

	dimensions = output.at(1).dims();
	EXPECT_EQ(2, dimensions);
	EXPECT_EQ(2, output.at(1).dim_size(0));
	EXPECT_EQ(2, output.at(1).dim_size(1));
	std::cout << output.at(1).DebugString() << std::endl;

	/*TTypes<int>::Matrix mat =  output.at(1).matrix<int>();
	::std::cout << mat << ::std::endl;*/
}