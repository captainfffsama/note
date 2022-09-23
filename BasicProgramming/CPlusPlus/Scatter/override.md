#CPP 

>C++17入门经典 P361

由于子类重写虚函数要求虚[函数签名](函数签名.md)和基类定义虚函数签名一模一样,容易出错.因此可以在子类重写的虚函数声明末尾加上关键字 `override`,如下:  
```cpp
class Son :public father
{
public:
	// father 中有一个虚函数 virtual double fun() const;
	double fun() const override
	{
		//...
	}
};
```

`override` 限定符会告诉编译器检查基类是否用了同样的签名声明来一个虚成员,若没有,编译器报错.  
**注意 `override` 和 `[virtual](virtual.md)` 限定符类似,它仅能出现在类定义中,不能用于成员函数的外部定义.**  

**在虚函数重写的声明中总是应该添加 `override` 限定符!** 但是在子类重写的虚函数中添加 `virtual` 是非必要的,取决于编程风格指南.

**注意 `override` 并不是关键字,因为把它作为关键字会破坏之前的代码. 技术上确实可以使用 `override` 作为变量甚至类的名称,但是最好不要这样做!**