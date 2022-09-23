#CPP 

>C++17 入门经典 P362

希望在子类中禁止重写某个成员函数,可以在其声明末尾加上 `final`.[这里末尾若同时存在 `final` , override,`const`, `final`和`override`须在const之后,但`final`和`override`顺序无所谓
](函数签名.md#^3da7c9)   
也可以把类指定为 final,来禁止继承,但是注意不要在 final 类中引入新的虚函数
```cpp
// Son 不能被继承
class Son final : public father
{
	//...
}
```

**注意 `final` 并不是关键字,因为把它作为关键字会破坏之前的代码. 技术上确实可以使用 `final` 作为变量甚至类的名称,但是最好不要这样做!**