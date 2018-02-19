#include "gtest/gtest.h"
#include "testhelper.h"

#include <vector>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"	// for placeholder
#include "tensorflow/cc/ops/math_ops.h"		// for Add, Abs, etc
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/state_ops.h" //For Variable

using namespace tensorflow;

class SaveAndRestore : public ::testing::Test
{
protected:
	Scope root = Scope::NewRootScope();
	ClientSession session = root;
	Input var = ops::Variable(root.WithOpName("variable"), { 1 }, DT_INT32);
	ops::Placeholder input = ops::Placeholder(root.WithOpName("input"), DT_INT32);

	void SetUp() override
	{
		::std::vector<Tensor> output;
		Output varAssign = ops::Assign(root.WithOpName("variableAssign"), var, { 1 });

		Status st{ session.Run({ varAssign }, &output) };
		ASSERT_TRUE(st.ok()) << st.error_message();
		ASSERT_EQ(1, *output.at(0).scalar<int>().data());
		output.clear();
	}

	void TearDown() override
	{
	}
};

TEST_F(SaveAndRestore, Variable)
{	
	GraphDef graphDef;
	root.ToGraphDef(&graphDef);

	Status st = WriteTextProto(Env::Default(), "SaveAndRestore-Variable.pbtext", graphDef);
	ASSERT_TRUE(st.ok()) << st.error_message();
}

TEST_F(SaveAndRestore, DisplayNodes)
{
	GraphDef graphDef;
	root.ToGraphDef(&graphDef);

	#define  NODE(idx) graphDef.node(idx)

	for (int idx = 0; idx < graphDef.node_size(); idx++)
	{
		std::cout << NODE(idx).name() << "/t" << NODE(idx).op() << std::endl;
	}
}


#include "tensorflow/cc/ops/io_ops.h" //For SaveV2

TEST_F(SaveAndRestore, SaveV2)
{
	auto add = ops::Add(root, var, input);

	InputList oLst{ var };
	string filename = "./ckpt";
	string names = "variable";
	string shapes = "";
	Tensor prefix(DT_STRING, TensorShape({}));
	prefix.scalar<string>()() = filename;
	Tensor shapeAndSlices(DT_STRING, TensorShape({1}));
	shapeAndSlices.scalar<string>()() = shapes;
	Tensor tensor_names(DT_STRING, TensorShape({1}));
	tensor_names.scalar<string>()() = names;

	//Operation Saver = ops::SaveV2(root, filename, names, shapes, oLst);
	Operation Saver = ops::SaveV2(root, prefix, tensor_names, shapeAndSlices, oLst);

	std::vector<Operation> out{ Saver };
	Status st{ session.Run({ { input,{3} }},{},out,nullptr) };
	ASSERT_TRUE(st.ok()) << st.error_message();
}