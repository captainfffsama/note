#C11
#CPP 

[toc]
 
# 简述
最低要求 C++ 11..它会构建一个 `tuple` 对象,这个对象中每个元素都应用自`tie` 传入的参数,并且引用的顺序是和参数顺序是一致的..  
常被用于`tuple`的解包.  使用`std::ignore` 可以忽略需要解包的`tuple`中的一些参数.
 
函数签名是:
```cpp
 template<class... Types>
  tuple<Types&...> tie (Types&... args) noexcept;
```
 
# 使用示例
```cpp
// packing/unpacking tuples
#include <iostream>     // std::cout
#include <tuple>        // std::tuple, std::make_tuple, std::tie

int main ()
{
  int myint;
  char mychar;

  std::tuple<int,float,char> mytuple;

  mytuple = std::make_tuple (10, 2.6, 'a');          // packing values into tuple

  std::tie (myint, std::ignore, mychar) = mytuple;   // unpacking tuple into variables

  std::cout << "myint contains: " << myint << '\n';
  std::cout << "mychar contains: " << mychar << '\n';

  return 0;
}
```

**输出**:
>myint contains: 10   
mychar contains: a


# 参考
- https://stackoverflow.com/questions/43762651/how-does-stdtie-work  
- http://www.cplusplus.com/reference/tuple/tie/