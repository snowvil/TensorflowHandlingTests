#include "gtest/gtest.h"
#include "testhelper.h"

#include <vector>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h" //For Placeholder
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/state_ops.h" //For Variable

#include "tensorflow/cc/ops/training_ops.h" // For ApplyGradientsDecent

using namespace ::tensorflow;

TEST(Training, ApplyGradientsDescent)
{
	TensorflowWrapperInit;
	Output variable = ops::Variable(root, { 1 }, DT_FLOAT);
	Output assign = ops::Assign(root, variable, { 1.0F });

	ClientSession sess(root);
	Status status({ sess.Run({assign},nullptr) });

	// if the documentation is talking about scalar, it seems, that the function does not need 
	// an initializer list for that value
	Output applyGradDescent = ops::ApplyGradientDescent(root, variable, 0.1F, { 1.0F });
	status = sess.Run({}, { applyGradDescent }, &output);
	EXPECT_TRUE(status.ok()) << status.error_message();
	float* val = output.at(0).scalar<float>().data();
	EXPECT_FLOAT_EQ(0.9, *val);
}