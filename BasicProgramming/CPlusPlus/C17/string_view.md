#CPP 
#C17

# 使用准则
1. 总是为输入参数使用 `std::string_view` 而不是 `const std::string&` .虽然使用 `const std::string_view&` 也没有问题,但是那样还不如按值传递 `std::string_view`, 因为复制这些对象的成本很低.
	>因为使用 `const std::string&` 作为参数,那么在函数调用时直接传入一个字符串字面量,其类型是`const char[]`,编译器依然会产生一个临时的`std:string`对其进行隐式转换.而使用`std::string_view`就没有这种性能浪费了
 ^c2f1b9
2.

# 参考
- C++17入门经典 第5版 P192 8.3.3节 