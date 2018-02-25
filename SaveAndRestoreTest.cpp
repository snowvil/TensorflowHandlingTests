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
		Output Multiplay = ops::Mul(root.WithOpName("multiply"), var, input);

		Status st{ session.Run({ varAssign }, &output) };
		ASSERT_TRUE(st.ok()) << st.error_message();
		ASSERT_EQ(1, *output.at(0).scalar<int>().data());
		output.clear();
	}

	void TearDown() override
	{
	}
};

TEST_F(SaveAndRestore, WriteTextProto)
{	
	GraphDef graphDef;
	root.ToGraphDef(&graphDef);

	Status st = WriteTextProto(Env::Default(), "./SaveAndRestore-Variable.pbtext", graphDef);
	ASSERT_TRUE(st.ok()) << st.error_message();
}

TEST_F(SaveAndRestore, WriteBinaryProto)
{
	GraphDef graphDef;
	root.ToGraphDef(&graphDef);

	Status st = WriteBinaryProto(Env::Default(), "./SaveAndRestore-Variable.pb", graphDef);
	ASSERT_TRUE(st.ok()) << st.error_message();
}

#include <experimental/filesystem>
#include "tensorflow/core/public/session.h"

TEST(Reading, ReadBinaryProto)
{
	std::string filename{ "./SaveAndRestore-Variable.pb" };
	ASSERT_TRUE(std::experimental::filesystem::exists(filename));

	GraphDef graphDef;	
	Status st = ReadBinaryProto(Env::Default(), filename, &graphDef);
	ASSERT_TRUE(st.ok()) << st.error_message();

	tensorflow::SessionOptions options;
	std::unique_ptr<tensorflow::Session>
	session(tensorflow::NewSession(options));
	tensorflow::Status s = session->Create(graphDef);

	// some typedefs for better handling
	typedef std::pair<string, Tensor> INPUT;
	std::vector<INPUT> oVec;

	Tensor inputVal{ DT_INT32,TensorShape({1}) };
	inputVal.scalar<int>()() = 2;
	INPUT oInputPlaceholder{ "input", inputVal };
	oVec.push_back(oInputPlaceholder);

	// have to manually call the variable assignation
	st = session->Run({}, { "variableAssign" }, {}, nullptr);
	ASSERT_TRUE(st.ok()) << st.error_message();

	// call the graph
	::std::vector<Tensor> output;
	st = session->Run(oVec, { "multiply" }, {}, &output);
	ASSERT_TRUE(st.ok()) << st.error_message();
	EXPECT_EQ(output.at(0).scalar<int>()(), 2);
	session->Close();
}

TEST_F(SaveAndRestore, DisplayNodes)
{
	GraphDef graphDef;
	root.ToGraphDef(&graphDef);

	#define  NODE(idx) graphDef.node(idx)

	for (int idx = 0; idx < graphDef.node_size(); idx++)
	{
		std::cout << NODE(idx).name() << "\t" << NODE(idx).op() << std::endl;
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

	Operation Saver = ops::SaveV2(root, prefix, tensor_names, shapeAndSlices, oLst);

	std::vector<Operation> out{ Saver };
	Status st = session.Run({},{},out,nullptr);
	ASSERT_TRUE(st.ok()) << st.error_message();
}

TEST(Restore, RestoreV2)
{
	Scope root{ Scope::NewRootScope() };

	string filename = "./ckpt";
	string names = "variable";
	string shapes = "";

	Tensor prefix(DT_STRING, TensorShape({}));
	prefix.scalar<string>()() = filename;
	Tensor shapeAndSlices(DT_STRING, TensorShape({ 1 }));
	shapeAndSlices.scalar<string>()() = shapes;
	Tensor tensor_names(DT_STRING, TensorShape({ 1 }));
	tensor_names.scalar<string>()() = names;

	DataTypeSlice dType{ DT_INT32 };

	ops::RestoreV2 Restore = ops::RestoreV2(root, { prefix }, { shapeAndSlices }, { shapeAndSlices }, dType);
}