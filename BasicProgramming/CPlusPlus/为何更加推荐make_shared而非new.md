#CPP 
#智能指针

参见 C++ primer plus 5th P413:
	因为 `make_shared` 会在对象分配的同时就将其与 `shared_ptr` 进行绑定,从而避免无意中将同一块内存绑定到多个独立创建的shared_ptr上.