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

TEST(Variable, VariableFloat)
{
	Scope root = Scope::NewRootScope();
	::std::vector<Tensor> output;

	ops::Variable var = ops::Variable(root, { 1 }, DT_FLOAT);
	Output varAssign = ops::Assign(root, var, { 1.0F });

	ClientSession session(root);
	Status st{ session.Run({ varAssign }, &output) };
	EXPECT_EQ(1.0F, *output.at(0).scalar<float>().data()) << st.error_message();
}

TEST(Variable, VariableIntAdd)
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

TEST(Variable, VariableFloatAdd)
{
	TensorflowWrapperInit;
	ops::Variable var = ops::Variable(root, { 1 }, DT_FLOAT);
	Output varAssign = ops::Assign(root, var, { 1.2F });

	ClientSession session(root);
	Status st{ session.Run({ varAssign }, &output) };
	EXPECT_EQ(1.2F, *output.at(0).scalar<float>().data());
	output.clear();

	Output add = ops::Add(root, var, ops::Cast(root,input,DT_FLOAT));
	st = session.Run({ { { input,{ 2 } } } }, { add }, &output);
	EXPECT_EQ(3.2F, *output.at(0).scalar<float>().data());
}

TEST(Variable, ManipulateVariable)
{
	// Adapting the multiplication value for input 'x' to output 'y'´.
	// This is some kind of 'Hello World in adapting'
	int x = 2;
	int y = 5;
	float weight = 1.0;

	TensorflowWrapperInit;
	ops::Placeholder y_expected = ops::Placeholder(root.WithOpName("y_expected"), DT_INT32);

	ops::Variable var = ops::Variable(root, { 1 }, DT_FLOAT);
	Output varAssign = ops::Assign(root, var, { weight });

	ClientSession session(root);
	Status st{ session.Run({ varAssign }, &output) };
	EXPECT_EQ(weight, *output.at(0).scalar<float>().data());
	output.clear();

	// adding the values. This is the basic model, which will be later used
	// to get the calculated values
	Output MulModel = ops::Mul(root, var, ops::Cast(root, input, DT_FLOAT));

	// calculating the difference between calculated value (output from 'add') and
	// the expected value 'y_expected'
	Output diff = ops::Sub(root, ops::Cast(root, y_expected, DT_FLOAT), MulModel);

	/*st = session.Run({ { { input,{ x } },{ y_expected,{y} } } }, { diff }, &output);
	float oDiff = *(output.at(0).scalar<float>().data());
	EXPECT_EQ(1.0, oDiff);
	output.clear();*/

	// multiplicate the obtained difference with a learning rate (0.1F)
	ops::Mul mul = ops::Mul(root, diff, { 0.1F });
	// add the learning value to the variable and assign the new value to the variable
	Output updateVariable = ops::Add(root, var, mul);
	Output varAssignNew = ops::Assign(root, var, updateVariable);

	::std::cout << "\tWeight\tOutput\n";
	// calculate the value 1000 times
	for (size_t i = 0; i < 1000; i++)
	{
		st = session.Run({ { { input,{ x } },{ y_expected,{ y } } } }, { varAssignNew,MulModel }, &output);
		// just to see the change of the variable
		if ((i < 20))
		{
			::std::cout<< i << "\t:" << *(output.at(0).scalar<float>().data()) << "\t" 
				<< *(output.at(1).scalar<float>().data()) << "\n";
		}
	}

	// see if the real model gets the desired value
	st = session.Run({ { { input,{ x } } } }, { MulModel }, &output);
	float oModelOutput = *(output.at(0).scalar<float>().data());
	EXPECT_FLOAT_EQ(y, oModelOutput);
}