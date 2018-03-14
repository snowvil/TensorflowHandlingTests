#define TensorflowWrapperInit \
	Scope root = Scope::NewRootScope(); \
	ops::Placeholder input = ops::Placeholder(root.WithOpName("input"), DT_INT32); \
	::std::vector<Tensor> output;\
	ClientSession session(root);


#define CHECK_STATUS(st) ASSERT_TRUE(st.ok()) << st.error_message()