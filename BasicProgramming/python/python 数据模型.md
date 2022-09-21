#python 
#待处理

[toc]



# 3. 数据模型[¶](https://docs.python.org/zh-cn/3/reference/datamodel.html#data-model)



## 3.1. 对象、值与类型

*对象* 是 Python 中对数据的抽象。 Python 程序中的所有数据都是由对象或对象间关系来表示的。 （从某种意义上说，按照冯·诺依曼的“存储程序计算机”模型，代码本身也是由对象来表示的。）



每个对象都有各自的编号、类型和值。一个对象被创建后，它的 *编号* 就绝不会改变；你可以将其理解为该对象在内存中的地址。 '[`is`](https://docs.python.org/zh-cn/3/reference/expressions.html#is)' 运算符可以比较两个对象的编号是否相同；[`id()`](https://docs.python.org/zh-cn/3/library/functions.html#id) 函数能返回一个代表其编号的整型数。

**CPython implementation detail:** 在 CPython 中，`id(x)` 就是存放 `x` 的内存的地址。

对象的类型决定该对象所支持的操作 (例如 "对象是否有长度属性？") 并且定义了该类型的对象可能的取值。[`type()`](https://docs.python.org/zh-cn/3/library/functions.html#type) 函数能返回一个对象的类型 (类型本身也是对象)。与编号一样，一个对象的 *类型* 也是不可改变的。[1](https://docs.python.org/zh-cn/3/reference/datamodel.html#id8)

有些对象的 *值* 可以改变。值可以改变的对象被称为 *可变的*；值不可以改变的对象就被称为 *不可变的*。(一个不可变容器对象如果包含对可变对象的引用，当后者的值改变时，前者的值也会改变；但是该容器仍属于不可变对象，因为它所包含的对象集是不会改变的。因此，不可变并不严格等同于值不能改变，实际含义要更微妙。) 一个对象的可变性是由其类型决定的；例如，数字、字符串和元组是不可变的，而字典和列表是可变的。

对象绝不会被显式地销毁；然而，当无法访问时它们可能会被作为垃圾回收。允许具体的实现推迟垃圾回收或完全省略此机制 --- 如何实现垃圾回收是实现的质量问题，只要可访问的对象不会被回收即可。

**CPython implementation detail:** CPython 目前使用带有 (可选) 延迟检测循环链接垃圾的引用计数方案，会在对象不可访问时立即回收其中的大部分，但不保证回收包含循环引用的垃圾。请查看 [`gc`](https://docs.python.org/zh-cn/3/library/gc.html#module-gc) 模块的文档了解如何控制循环垃圾的收集相关信息。其他实现会有不同的行为方式，CPython 现有方式也可能改变。不要依赖不可访问对象的立即终结机制 (所以你应当总是显式地关闭文件)。

注意：使用实现的跟踪或调试功能可能令正常情况下会被回收的对象继续存活。还要注意通过 '[`try`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#try)...[`except`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#except)' 语句捕捉异常也可能令对象保持存活。

有些对象包含对 "外部" 资源的引用，例如打开文件或窗口。当对象被作为垃圾回收时这些资源也应该会被释放，但由于垃圾回收并不确保发生，这些对象还提供了明确地释放外部资源的操作，通常为一个 `close()` 方法。强烈推荐在程序中显式关闭此类对象。'[`try`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#try)...[`finally`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#finally)' 语句和 '[`with`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#with)' 语句提供了进行此种操作的更便捷方式。

有些对象包含对其他对象的引用；它们被称为 *容器*。容器的例子有元组、列表和字典等。这些引用是容器对象值的组成部分。在多数情况下，当谈论一个容器的值时，我们是指所包含对象的值而不是其编号；但是，当我们谈论一个容器的可变性时，则仅指其直接包含的对象的编号。因此，如果一个不可变容器 (例如元组) 包含对一个可变对象的引用，则当该可变对象被改变时容器的值也会改变。


类型会影响对象行为的几乎所有方面。甚至对象编号的重要性也在某种程度上受到影响: 对于不可变类型，会得出新值的运算实际上会返回对相同类型和取值的任一现有对象的引用，而对于可变类型来说这是不允许的。例如在 `a = 1; b = 1` 之后，`a` 和 `b` 可能会也可能不会指向同一个值为一的对象，这取决于具体实现，但是在 `c = []; d = []` 之后，`c` 和 `d` 保证会指向两个不同、单独的新建空列表。(请注意 `c = d = []` 则是将同一个对象赋值给 `c` 和 `d`。)




## 3.2. 标准类型层级结构

以下是 Python 内置类型的列表。扩展模块 (具体实现会以 C, Java 或其他语言编写) 可以定义更多的类型。未来版本的 Python 可能会加入更多的类型 (例如有理数、高效存储的整型数组等等)，不过新增类型往往都是通过标准库来提供的。

以下部分类型的描述中包含有 '特殊属性列表' 段落。这些属性提供对具体实现的访问而非通常使用。它们的定义在未来可能会改变。

- None

    此类型只有一种取值。是一个具有此值的单独对象。此对象通过内置名称 `None` 访问。在许多情况下它被用来表示空值，例如未显式指明返回值的函数将返回 None。它的逻辑值为假。

- NotImplemented

    此类型只有一种取值。是一个具有此值的单独对象。此对象通过内置名称 `NotImplemented` 访问。数值方法和丰富比较方法如未实现指定运算符表示的运算则应返回此值。(解释器会根据指定运算符继续尝试反向运算或其他回退操作)。它的逻辑值为真。详情参见 [实现算数运算](https://docs.python.org/zh-cn/3/library/numbers.html#implementing-the-arithmetic-operations)。

- Ellipsis

    此类型只有一种取值。是一个具有此值的单独对象。此对象通过字面值 `...` 或内置名称 `Ellipsis` 访问。它的逻辑值为真。

- [`numbers.Number`](https://docs.python.org/zh-cn/3/library/numbers.html#numbers.Number)

    此类对象由数字字面值创建，并会被作为算术运算符和算术内置函数的返回结果。数字对象是不可变的；一旦创建其值就不再改变。Python 中的数字当然非常类似数学中的数字，但也受限于计算机中的数字表示方法。
	
	Python 区分整型数、浮点型数和复数:
	- [`numbers.Integral`](https://docs.python.org/zh-cn/3/library/numbers.html#numbers.Integral)此类对象表示数学中整数集合的成员 (包括正数和负数)。
		整型数可细分为两种类型:
		- 整型 ([`int`](https://docs.python.org/zh-cn/3/library/functions.html#int))
			此类对象表示任意大小的数字，仅受限于可用的内存 (包括虚拟内存)。在变换和掩码运算中会以二进制表示，负数会以 2 的补码表示，看起来像是符号位向左延伸补满空位。
		- 布尔型 ([`bool`](https://docs.python.org/zh-cn/3/library/functions.html#bool))
			此类对象表示逻辑值 False 和 True。代表 `False` 和 `True` 值的两个对象是唯二的布尔对象。布尔类型是整型的子类型，两个布尔值在各种场合的行为分别类似于数值 0 和 1，例外情况只有在转换为字符串时分别返回字符串 `"False"` 或 `"True"`。整型数表示规则的目的是在涉及负整型数的变换和掩码运算时提供最为合理的解释。
			
	- [`numbers.Real`](https://docs.python.org/zh-cn/3/library/numbers.html#numbers.Real) ([`float`](https://docs.python.org/zh-cn/3/library/functions.html#float))
		此类对象表示机器级的双精度浮点数。其所接受的取值范围和溢出处理将受制于底层的机器架构 (以及 C 或 Java 实现)。Python 不支持单精度浮点数；支持后者通常的理由是节省处理器和内存消耗，但这点节省相对于在 Python 中使用对象的开销来说太过微不足道，因此没有理由包含两种浮点数而令该语言变得复杂。
		
	- [`numbers.Complex`](https://docs.python.org/zh-cn/3/library/numbers.html#numbers.Complex) ([`complex`](https://docs.python.org/zh-cn/3/library/functions.html#complex))
		此类对象以一对机器级的双精度浮点数来表示复数值。有关浮点数的附带规则对其同样有效。一个复数值 `z` 的实部和虚部可通过只读属性 `z.real` 和 `z.imag` 来获取。

- 序列

    此类对象表示以非负整数作为索引的有限有序集。内置函数 [`len()`](https://docs.python.org/zh-cn/3/library/functions.html#len) 可返回一个序列的条目数量。当一个序列的长度为 *n* 时，索引集包含数字 0, 1, ..., *n*-1。序列 *a* 的条目 *i* 可通过 `a[i]` 选择。
	
	序列还支持切片: `a[i:j]` 选择索引号为 *k* 的所有条目，*i* `<=` *k* `<` *j*。当用作表达式时，序列的切片就是一个与序列类型相同的新序列。新序列的索引还是从 0 开始。
	
	有些序列还支持带有第三个 "step" 形参的 "扩展切片": `a[i:j:k]` 选择 *a* 中索引号为 *x* 的所有条目，`x = i + n*k`, *n* `>=` `0` 且 *i* `<=` *x* `<` *j*。
	
	序列可根据其可变性来加以区分:
	- 不可变序列
		不可变序列类型的对象一旦创建就不能再改变。(如果对象包含对其他对象的引用，其中的可变对象就是可以改变的；但是，一个不可变对象所直接引用的对象集是不能改变的。)
		以下类型属于不可变对象:
		- 字符串
			字符串是由 Unicode 码位值组成的序列。范围在 `U+0000 - U+10FFFF` 之内的所有码位值都可在字符串中使用。Python 没有 `char` 类型；而是将字符串中的每个码位表示为一个长度为 `1` 的字符串对象。内置函数 [`ord()`](https://docs.python.org/zh-cn/3/library/functions.html#ord) 可将一个码位由字符串形式转换成一个范围在 `0 - 10FFFF` 之内的整型数；[`chr()`](https://docs.python.org/zh-cn/3/library/functions.html#chr) 可将一个范围在 `0 - 10FFFF` 之内的整型数转换为长度为 `1` 的对应字符串对象。[`str.encode()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#str.encode) 可以使用指定的文本编码将 [`str`](https://docs.python.org/zh-cn/3/library/stdtypes.html#str) 转换为 [`bytes`](https://docs.python.org/zh-cn/3/library/stdtypes.html#bytes)，而 [`bytes.decode()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#bytes.decode) 则可以实现反向的解码。
		- 元组
			一个元组中的条目可以是任意 Python 对象。包含两个或以上条目的元组由逗号分隔的表达式构成。只有一个条目的元组 ('单项元组') 可通过在表达式后加一个逗号来构成 (一个表达式本身不能创建为元组，因为圆括号要用来设置表达式分组)。一个空元组可通过一对内容为空的圆括号创建。字节串字节串对象是不可变的数组。其中每个条目都是一个 8 位字节，以取值范围 0 <= x < 256 的整型数表示。字节串字面值 (例如 `b'abc'`) 和内置的 [`bytes()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#bytes) 构造器可被用来创建字节串对象。字节串对象还可以通过 [`decode()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#bytes.decode) 方法解码为字符串。
			
	- 可变序列
		可变序列在被创建后仍可被改变。下标和切片标注可被用作赋值和 [`del`](https://docs.python.org/zh-cn/3/reference/simple_stmts.html#del) (删除) 语句的目标。
		
		目前有两种内生可变序列类型:
		- 列表
			列表中的条目可以是任意 Python 对象。列表由用方括号括起并由逗号分隔的多个表达式构成。(注意创建长度为 0 或 1 的列表无需使用特殊规则。)
			
		- 字节数组
			字节数组对象属于可变数组。可以通过内置的 [`bytearray()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#bytearray) 构造器来创建。除了是可变的 (因而也是不可哈希的)，在其他方面字节数组提供的接口和功能都于不可变的 [`bytes`](https://docs.python.org/zh-cn/3/library/stdtypes.html#bytes) 对象一致。
		
		扩展模块 [`array`](https://docs.python.org/zh-cn/3/library/array.html#module-array) 提供了一个额外的可变序列类型示例，[`collections`](https://docs.python.org/zh-cn/3/library/collections.html#module-collections) 模块也是如此。

- 集合类型

    此类对象表示由不重复且不可变对象组成的无序且有限的集合。因此它们不能通过下标来索引。但是它们可被迭代，也可用内置函数 [`len()`](https://docs.python.org/zh-cn/3/library/functions.html#len) 返回集合中的条目数。集合常见的用处是快速成员检测，去除序列中的重复项，以及进行交、并、差和对称差等数学运算。
	
	对于集合元素所采用的不可变规则与字典的键相同。注意数字类型遵循正常的数字比较规则: 如果两个数字相等 (例如 `1` 和 `1.0`)，则同一集合中只能包含其中一个。
	
	目前有两种内生集合类型:
	
	- 集合
		此类对象表示可变集合。它们可通过内置的 [`set()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#set) 构造器创建，并且创建之后可以通过方法进行修改，例如 `add()`。
	
	- 冻结集合
		此类对象表示不可变集合。它们可通过内置的 [`frozenset()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#frozenset) 构造器创建。由于 frozenset 对象不可变且 [hashable](https://docs.python.org/zh-cn/3/glossary.html#term-hashable)，它可以被用作另一个集合的元素或是字典的键。

- 映射

    此类对象表示由任意索引集合所索引的对象的集合。通过下标 `a[k]` 可在映射 `a` 中选择索引为 `k` 的条目；这可以在表达式中使用，也可作为赋值或 [`del`](https://docs.python.org/zh-cn/3/reference/simple_stmts.html#del) 语句的目标。内置函数 [`len()`](https://docs.python.org/zh-cn/3/library/functions.html#len) 可返回一个映射中的条目数。
	目前只有一种内生映射类型:
	
	- 字典
		此类对象表示由几乎任意值作为索引的有限个对象的集合。不可作为键的值类型只有包含列表或字典或其他可变类型，通过值而非对象编号进行比较的值，其原因在于高效的字典实现需要使用键的哈希值以保持一致性。用作键的数字类型遵循正常的数字比较规则: 如果两个数字相等 (例如 `1` 和 `1.0`) 则它们均可来用来索引同一个字典条目。
		
		字典会保留插入顺序，这意味着键将以它们被添加的顺序在字典中依次产生。 替换某个现有的键不会改变其顺序，但是移除某个键再重新插入则会将其添加到末尾而不会保留其原有位置。
		
		字典是可变的；它们可通过 `{...}` 标注来创建 (参见 [字典显示](https://docs.python.org/zh-cn/3/reference/expressions.html#dict) 小节)。
		
		扩展模块 [`dbm.ndbm`](https://docs.python.org/zh-cn/3/library/dbm.html#module-dbm.ndbm) 和 [`dbm.gnu`](https://docs.python.org/zh-cn/3/library/dbm.html#module-dbm.gnu) 提供了额外的映射类型示例，[`collections`](https://docs.python.org/zh-cn/3/library/collections.html#module-collections) 模块也是如此。
		
		*在 3.7 版更改:* 在 Python 3.6 版之前字典不会保留插入顺序。 在 CPython 3.6 中插入顺序会被保留，但这在当时被当作是一个实现细节而非确定的语言特性。

- 可调用类型

    此类型可以被应用于函数调用操作 (参见 [调用](https://docs.python.org/zh-cn/3/reference/expressions.html#calls) 小节):
	- 用户定义函数
		用户定义函数对象可通过函数定义来创建 (参见 [函数定义](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#function) 小节)。它被调用时应附带一个参数列表，其中包含的条目应与函数所定义的形参列表一致。
		
		特殊属性:
		
		属性含义`__doc__`该函数的文档字符串，没有则为 `None`；不会被子类继承。可写[`__name__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#definition.__name__)该函数的名称。可写[`__qualname__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#definition.__qualname__)该函数的 [qualified name](https://docs.python.org/zh-cn/3/glossary.html#term-qualified-name)。*3.3 新版功能.*可写`__module__`该函数所属模块的名称，没有则为 `None`。可写`__defaults__`由具有默认值的参数的默认参数值组成的元组，如无任何参数具有默认值则为 `None`。可写`__code__`表示编译后的函数体的代码对象。可写`__globals__`对存放该函数中全局变量的字典的引用 --- 函数所属模块的全局命名空间。只读[`__dict__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#object.__dict__)命名空间支持的函数属性。可写`__closure__``None` 或包含该函数可用变量的绑定的单元的元组。有关 `cell_contents` 属性的详情见下。只读`__annotations__`包含参数标注的字典。字典的键是参数名，如存在返回标注则为 `'return'`。可写`__kwdefaults__`仅包含关键字参数默认值的字典。
		
		可写大部分标有 "Writable" 的属性均会检查赋值的类型。
		函数对象也支持获取和设置任意属性，例如这可以被用来给函数附加元数据。使用正规的属性点号标注获取和设置此类属性。*注意当前实现仅支持用户定义函数属性。未来可能会增加支持内置函数属性。*
		
		单元对象具有 `cell_contents` 属性。这可被用来获取以及设置单元的值。
		
		有关函数定义的额外信息可以从其代码对象中提取；参见下文对内部类型的描述。 [`cell`](https://docs.python.org/zh-cn/3/library/types.html#types.CellType) 类型可以在 [`types`](https://docs.python.org/zh-cn/3/library/types.html#module-types) 模块中访问。
		
	- 实例方法
		实例方法用于结合类、类实例和任何可调用对象 (通常为用户定义函数)。
		
		特殊的只读属性: `__self__` 为类实例对象本身，`__func__` 为函数对象；`__doc__` 为方法的文档 (与 `__func__.__doc__` 作用相同)；[`__name__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#definition.__name__) 为方法名称 (与 `__func__.__name__` 作用相同)；`__module__` 为方法所属模块的名称，没有则为 `None`。
		
		方法还支持获取 (但不能设置) 下层函数对象的任意函数属性。
		
		用户定义方法对象可在获取一个类的属性时被创建 (也可能通过该类的一个实例)，如果该属性为用户定义函数对象或类方法对象。
		
		当通过从类实例获取一个用户定义函数对象的方式创建一个实例方法对象时，类实例对象的 `__self__` 属性即为该实例，并会绑定方法对象。该新建方法的 `__func__` 属性就是原来的函数对象。
		
		当通过从类或实例获取一个类方法对象的方式创建一个实例对象时，实例对象的 `__self__` 属性为该类本身，其 `__func__` 属性为类方法对应的下层函数对象。
		
		当一个实例方法对象被调用时，会调用对应的下层函数 (`__func__`)，并将类实例 (`__self__`) 插入参数列表的开头。例如，当 `C` 是一个包含了 `f()` 函数定义的类，而 `x` 是 `C` 的一个实例，则调用 `x.f(1)` 就等同于调用 `C.f(x, 1)`。
		
		当一个实例方法对象是衍生自一个类方法对象时，保存在 `__self__` 中的 "类实例" 实际上会是该类本身，因此无论是调用 `x.f(1)` 还是 `C.f(1)` 都等同于调用 `f(C,1)`，其中 `f` 为对应的下层函数。
		
		请注意从函数对象到实例方法对象的变换会在每一次从实例获取属性时发生。在某些情况下，一种高效的优化方式是将属性赋值给一个本地变量并调用该本地变量。还要注意这样的变换只发生于用户定义函数；其他可调用对象 (以及所有不可调用对象) 在被获取时都不会发生变换。还有一个需要关注的要点是作为一个类实例属性的用户定义函数不会被转换为绑定方法；这样的变换 *仅当* 函数是类属性时才会发生。
		
	- 生成器函数
		一个使用 [`yield`](https://docs.python.org/zh-cn/3/reference/simple_stmts.html#yield) 语句 （见 [yield 语句](https://docs.python.org/zh-cn/3/reference/simple_stmts.html#yield) 章节）的函数或方法被称作一个 *生成器函数*。 这样的函数在被调用时，总是返回一个可以执行函数体的迭代器对象：调用该迭代器的 [`iterator.__next__()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#iterator.__next__) 方法将会导致这个函数一直运行直到它使用 `yield` 语句提供了一个值为止。 当这个函数执行 [`return`](https://docs.python.org/zh-cn/3/reference/simple_stmts.html#return) 语句或者执行到末尾时，将引发 [`StopIteration`](https://docs.python.org/zh-cn/3/library/exceptions.html#StopIteration) 异常并且这个迭代器将到达所返回的值集合的末尾。
		
	- 协程函数
		使用 [`async def`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#async-def) 来定义的函数或方法就被称为 *协程函数*。这样的函数在被调用时会返回一个 [coroutine](https://docs.python.org/zh-cn/3/glossary.html#term-coroutine) 对象。它可能包含 [`await`](https://docs.python.org/zh-cn/3/reference/expressions.html#await) 表达式以及 [`async with`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#async-with) 和 [`async for`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#async-for) 语句。详情可参见 [协程对象](https://docs.python.org/zh-cn/3/reference/datamodel.html#coroutine-objects) 一节。
		
	- 异步生成器函数
		使用 [`async def`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#async-def) 来定义并包含 [`yield`](https://docs.python.org/zh-cn/3/reference/simple_stmts.html#yield) 语句的函数或方法就被称为 *异步生成器函数*。这样的函数在被调用时会返回一个异步迭代器对象，该对象可在 [`async for`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#async-for) 语句中用来执行函数体。调用异步迭代器的 `aiterator.__anext__()` 方法将会返回一个 [awaitable](https://docs.python.org/zh-cn/3/glossary.html#term-awaitable)，此对象会在被等待时执行直到使用 [`yield`](https://docs.python.org/zh-cn/3/reference/simple_stmts.html#yield) 表达式输出一个值。当函数执行时到空的 [`return`](https://docs.python.org/zh-cn/3/reference/simple_stmts.html#return) 语句或是最后一条语句时，将会引发 [`StopAsyncIteration`](https://docs.python.org/zh-cn/3/library/exceptions.html#StopAsyncIteration) 异常，异步迭代器也会到达要输出的值集合的末尾。
		
	- 内置函数
		内置函数对象是对于 C 函数的外部封装。内置函数的例子包括 [`len()`](https://docs.python.org/zh-cn/3/library/functions.html#len) 和 [`math.sin()`](https://docs.python.org/zh-cn/3/library/math.html#math.sin) ([`math`](https://docs.python.org/zh-cn/3/library/math.html#module-math) 是一个标准内置模块)。内置函数参数的数量和类型由 C 函数决定。特殊的只读属性: `__doc__` 是函数的文档字符串，如果没有则为 `None`; [`__name__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#definition.__name__) 是函数的名称; `__self__` 设定为 `None` (参见下一条目); `__module__` 是函数所属模块的名称，如果没有则为 `None`。
		
	- 内置方法
		此类型实际上是内置函数的另一种形式，只不过还包含了一个传入 C 函数的对象作为隐式的额外参数。内置方法的一个例子是 `alist.append()`，其中 *alist* 为一个列表对象。在此示例中，特殊的只读属性 `__self__` 会被设为 *alist* 所标记的对象。
		
	- 类
		类是可调用的。此种对象通常是作为“工厂”来创建自身的实例，类也可以有重载 [`__new__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__new__) 的变体类型。调用的参数会传给 [`__new__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__new__)，而且通常也会传给 [`__init__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__init__) 来初始化新的实例。
		
	- 类实例
		任意类的实例通过在所属类中定义 [`__call__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__call__) 方法即能成为可调用的对象。

- 模块

    模块是 Python 代码的基本组织单元，由 [导入系统](https://docs.python.org/zh-cn/3/reference/import.html#importsystem) 创建，由 [`import`](https://docs.python.org/zh-cn/3/reference/simple_stmts.html#import) 语句发起调用，或者通过 [`importlib.import_module()`](https://docs.python.org/zh-cn/3/library/importlib.html#importlib.import_module) 和内置的 [`__import__()`](https://docs.python.org/zh-cn/3/library/functions.html#__import__) 等函数发起调用。 模块对象具有由字典对象实现的命名空间（这是被模块中定义的函数的 `__globals__` 属性引用的字典）。 属性引用被转换为该字典中的查找，例如 `m.x` 相当于 `m.__dict__["x"]`。 模块对象不包含用于初始化模块的代码对象（因为初始化完成后不需要它）。
	
	属性赋值会更新模块的命名空间字典，例如 `m.x = 1` 等同于 `m.__dict__["x"] = 1`。
	
	预定义的 (可写) 属性: [`__name__`](https://docs.python.org/zh-cn/3/reference/import.html#__name__) 为模块的名称; `__doc__` 为模块的文档字符串，如果没有则为 `None`; `__annotations__` (可选) 为一个包含 [变量标注](https://docs.python.org/zh-cn/3/glossary.html#term-variable-annotation) 的字典，它是在模块体执行时获取的; [`__file__`](https://docs.python.org/zh-cn/3/reference/import.html#__file__) 是模块对应的被加载文件的路径名，如果它是加载自一个文件的话。某些类型的模块可能没有 [`__file__`](https://docs.python.org/zh-cn/3/reference/import.html#__file__) 属性，例如 C 模块是静态链接到解释器内部的; 对于从一个共享库动态加载的扩展模块来说该属性为该共享库文件的路径名。
	
	特殊的只读属性: [`__dict__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#object.__dict__) 为以字典对象表示的模块命名空间。
	
	**CPython implementation detail:** 由于 CPython 清理模块字典的设定，当模块离开作用域时模块字典将会被清理，即使该字典还有活动的引用。想避免此问题，可复制该字典或保持模块状态以直接使用其字典。

- 自定义类

    自定义类这种类型一般通过类定义来创建 (参见 [类定义](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#class) 一节)。每个类都有通过一个字典对象实现的独立命名空间。类属性引用会被转化为在此字典中查找，例如 `C.x` 会被转化为 `C.__dict__["x"]` (不过也存在一些钩子对象以允许其他定位属性的方式)。当未在其中发现某个属性名称时，会继续在基类中查找。这种基类查找使用 C3 方法解析顺序，即使存在 '钻石形' 继承结构即有多条继承路径连到一个共同祖先也能保持正确的行为。有关 Python 使用的 C3 MRO 的详情可查看配合 2.3 版发布的文档 https://www.python.org/download/releases/2.3/mro/.当一个类属性引用 (假设类名为 `C`) 会产生一个类方法对象时，它将转化为一个 `__self__` 属性为 `C` 的实例方法对象。当其会产生一个静态方法对象时，它将转化为该静态方法对象所封装的对象。从类的 [`__dict__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#object.__dict__) 所包含内容以外获取属性的其他方式请参看 [实现描述器](https://docs.python.org/zh-cn/3/reference/datamodel.html#descriptors) 一节。
	
	类属性赋值会更新类的字典，但不会更新基类的字典。
	
	类对象可被调用 (见上文) 以产生一个类实例 (见下文)。
	
	特殊属性: [`__name__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#definition.__name__) 为类的名称; `__module__` 为类所在模块的名称; [`__dict__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#object.__dict__) 为包含类命名空间的字典; [`__bases__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#class.__bases__) 为包含基类的元组，按其在基类列表中的出现顺序排列; `__doc__` 为类的文档字符串，如果没有则为 `None`; `__annotations__` (可选) 为一个包含 [变量标注](https://docs.python.org/zh-cn/3/glossary.html#term-variable-annotation) 的字典，它是在类体执行时获取的。

- 类实例

    类实例可通过调用类对象来创建 (见上文)。每个类实例都有通过一个字典对象实现的独立命名空间，属性引用会首先在此字典中查找。当未在其中发现某个属性，而实例对应的类中有该属性时，会继续在类属性中查找。如果找到的类属性为一个用户定义函数对象，它会被转化为实例方法对象，其 `__self__` 属性即该实例。静态方法和类方法对象也会被转化；参见上文 "Classes" 一节。要了解其他通过类实例来获取相应类属性的方式可参见 [实现描述器](https://docs.python.org/zh-cn/3/reference/datamodel.html#descriptors) 一节，这样得到的属性可能与实际存放于类的 [`__dict__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#object.__dict__) 中的对象不同。如果未找到类属性，而对象对应的类具有 [`__getattr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getattr__) 方法，则会调用该方法来满足查找要求。
	
	属性赋值和删除会更新实例的字典，但不会更新对应类的字典。如果类具有 [`__setattr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__setattr__) 或 [`__delattr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__delattr__) 方法，则将调用方法而不再直接更新实例的字典。
	
	如果类实例具有某些特殊名称的方法，就可以伪装为数字、序列或映射。参见 [特殊方法名称](https://docs.python.org/zh-cn/3/reference/datamodel.html#specialnames) 一节。
	
	特殊属性: [`__dict__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#object.__dict__) 为属性字典; [`__class__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#instance.__class__) 为实例对应的类。

- I/O 对象 (或称文件对象)

    [file object](https://docs.python.org/zh-cn/3/glossary.html#term-file-object) 表示一个打开的文件。有多种快捷方式可用来创建文件对象: [`open()`](https://docs.python.org/zh-cn/3/library/functions.html#open) 内置函数，以及 [`os.popen()`](https://docs.python.org/zh-cn/3/library/os.html#os.popen), [`os.fdopen()`](https://docs.python.org/zh-cn/3/library/os.html#os.fdopen) 和 socket 对象的 [`makefile()`](https://docs.python.org/zh-cn/3/library/socket.html#socket.socket.makefile) 方法 (还可能使用某些扩展模块所提供的其他函数或方法)。
	
	`sys.stdin`, `sys.stdout` 和 `sys.stderr` 会初始化为对应于解释器标准输入、输出和错误流的文件对象；它们都会以文本模式打开，因此都遵循 [`io.TextIOBase`](https://docs.python.org/zh-cn/3/library/io.html#io.TextIOBase) 抽象类所定义的接口。

- 内部类型

    某些由解释器内部使用的类型也被暴露给用户。它们的定义可能随未来解释器版本的更新而变化，为内容完整起见在此处一并介绍。
	
	- 代码对象
		代码对象表示 *编译为字节的* 可执行 Python 代码，或称 [bytecode](https://docs.python.org/zh-cn/3/glossary.html#term-bytecode)。代码对象和函数对象的区别在于函数对象包含对函数全局对象 (函数所属的模块) 的显式引用，而代码对象不包含上下文；而且默认参数值会存放于函数对象而不是代码对象内 (因为它们表示在运行时算出的值)。与函数对象不同，代码对象不可变，也不包含对可变对象的引用 (不论是直接还是间接)。
		
		特殊的只读属性: `co_name` 给出了函数名; `co_argcount` 为位置参数的总数量 (包括仅限位置参数和带有默认值的参数); `co_posonlyargcount` 为仅限位置参数的数量 (包括带有默认值的参数); `co_kwonlyargcount` 为仅限关键字参数的数量 (包括带有默认值的参数); `co_nlocals` 为函数使用的局部变量的数量 (包括参数); `co_varnames` 为一个包含局部变量名称的元组 (参数名排在最前面); `co_cellvars` 为一个包含被嵌套函数所引用的局部变量名称的元组; `co_freevars` 为一个包含自由变量名称的元组; `co_code` 为一个表示字节码指令序列的字符口中; `co_consts` 为一个包含字节码所使用的字面值的元组; `co_names` 为一个包含字节码所使用的名称的元组; `co_filename` 为被编码代码所在的文件名; `co_firstlineno` 为函数首行的行号; `co_lnotab` 为一个字符串，其中编码了从字节码偏移量到行号的映射 (详情参见解释器的源代码); `co_stacksize` 为要求的栈大小; `co_flags` 为一个整数，其中编码了解释器所用的多个旗标。
		
		以下是可用于 `co_flags` 的标志位定义：如果函数使用 `*arguments` 语法来接受任意数量的位置参数，则 `0x04` 位被设置；如果函数使用 `**keywords` 语法来接受任意数量的关键字参数，则 `0x08` 位被设置；如果函数是一个生成器，则 `0x20` 位被设置。
		
		未来特性声明 (`from __future__ import division`) 也使用 `co_flags` 中的标志位来指明代码对象的编译是否启用特定的特性: 如果函数编译时启用未来除法特性则设置 `0x2000` 位; 在更早的 Python 版本中则使用 `0x10` 和 `0x1000` 位。
		
		`co_flags` 中的其他位被保留为内部使用。
		
		如果代码对象表示一个函数，`co_consts` 中的第一项将是函数的文档字符串，如果未定义则为 `None`。
		
	- 帧对象
		帧对象表示执行帧。它们可能出现在回溯对象中 (见下文)，还会被传递给注册跟踪函数。
		
		特殊的只读属性: `f_back` 为前一堆栈帧 (指向调用者)，如是最底层堆栈帧则为 `None`; `f_code` 为此帧中所执行的代码对象; `f_locals` 为用于查找本地变量的字典; `f_globals` 则用于查找全局变量; `f_builtins` 用于查找内置 (固有) 名称; `f_lasti` 给出精确指令 (这是代码对象的字节码字符串的一个索引)。
		
		特殊的可写属性: `f_trace`，如果不为 `None`，则是在代码执行期间调用各类事件的函数 (由调试器使用)。通常每个新源码行会触发一个事件 - 这可以通过将 `f_trace_lines` 设为 [`False`](https://docs.python.org/zh-cn/3/library/constants.html#False) 来禁用。
		
		具体的实现 *可能* 会通过将 `f_trace_opcodes` 设为 [`True`](https://docs.python.org/zh-cn/3/library/constants.html#True) 来允许按操作码请求事件。请注意如果跟踪函数引发的异常逃逸到被跟踪的函数中，这可能会导致未定义的解释器行为。
		
		`f_lineno` 为帧的当前行号 --- 在这里写入从一个跟踪函数内部跳转的指定行 (仅用于最底层的帧)。调试器可以通过写入 f_lineno 实现一个 Jump 命令 (即设置下一语句)。
		
		帧对象支持一个方法:
		`frame.clear`()
			此方法清除该帧持有的全部对本地变量的引用。而且如果该帧属于一个生成器，生成器会被完成。这有助于打破包含帧对象的循环引用 (例如当捕获一个异常并保存其回溯在之后使用)。
			如果该帧当前正在执行则会引发 [`RuntimeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#RuntimeError)。
			*3.4 新版功能.*
			
	- 回溯对象
		回溯对象表示一个异常的栈跟踪记录。当异常发生时会隐式地创建一个回溯对象，也可能通过调用 [`types.TracebackType`](https://docs.python.org/zh-cn/3/library/types.html#types.TracebackType) 显式地创建。
		
		对于隐式地创建的回溯对象，当查找异常句柄使得执行栈展开时，会在每个展开层级的当前回溯之前插入一个回溯对象。当进入一个异常句柄时，栈跟踪将对程序启用。(参见 [try 语句](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#try) 一节。) 它可作为 `sys.exc_info()` 所返回的元组的第三项，以及所捕获异常的 `__traceback__` 属性被获取。
		
		当程序不包含可用的句柄时，栈跟踪会 (以良好的格式) 写入标准错误流；如果解释器处于交互模式，它也可作为 `sys.last_traceback` 对用户启用。
		
		对于显式创建的回溯对象，则由回溯对象的创建者来决定应该如何链接 `tb_next` 属性来构成完整的栈跟踪。
		
		特殊的只读属性: `tb_frame` 指向当前层级的执行帧; `tb_lineno` 给出发生异常所在的行号; `tb_lasti` 标示具体指令。如果异常发生于没有匹配的 except 子句或有 finally 子句的 [`try`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#try) 语句中，回溯对象中的行号和最后指令可能与相应帧对象中行号不同。
		
		特殊的可写属性: `tb_next` 为栈跟踪中的下一层级 (通往发生异常的帧)，如果没有下一层级则为 `None`。
		
		*在 3.7 版更改:* 回溯对象现在可以使用 Python 代码显式地实例化，现有实例的 `tb_next` 属性可以被更新。
		
	- 切片对象
		切片对象用来表示 [`__getitem__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getitem__) 方法得到的切片。该对象也可使用内置的 [`slice()`](https://docs.python.org/zh-cn/3/library/functions.html#slice) 函数来创建。
		
		特殊的只读属性: `start` 为下界; `stop` 为上界; `step` 为步长值; 各值如省略则为 `None`。这些属性可具有任意类型。
		
		切片对象支持一个方法:
		`slice.indices`(*self*, *length*)
			此方法接受一个整型参数 *length* 并计算在切片对象被应用到 *length* 指定长度的条目序列时切片的相关信息应如何描述。其返回值为三个整型数组成的元组；这些数分别为切片的 *start* 和 *stop* 索引号以及 *step* 步长值。索引号缺失或越界则按照正规连续切片的方式处理。
			
	- 静态方法
		对象静态方法对象提供了一种避免上文所述将函数对象转换为方法对象的方式。静态方法对象为对任意其他对象的封装，通常用来封装用户定义方法对象。当从类或类实例获取一个静态方法对象时，实际返回的对象是封装的对象，它不会被进一步转换。静态方法对象自身不是可调用的，但它们所封装的对象通常都是可调用的。静态方法对象可通过内置的 [`staticmethod()`](https://docs.python.org/zh-cn/3/library/functions.html#staticmethod) 构造器来创建。
		
	- 类方法对象
		类方法对象和静态方法一样是对其他对象的封装，会改变从类或类实例获取该对象的方式。类方法对象在此类获取操作中的行为已在上文 "用户定义方法" 一节中描述。类方法对象可通过内置的 [`classmethod()`](https://docs.python.org/zh-cn/3/library/functions.html#classmethod) 构造器来创建。



## 3.3. 特殊方法名称

一个类可以通过定义具有特殊名称的方法来实现由特殊语法所引发的特定操作 (例如算术运算或下标与切片)。这是 Python 实现 *操作符重载* 的方式，允许每个类自行定义基于操作符的特定行为。例如，如果一个类定义了名为 [`__getitem__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getitem__) 的方法，并且 `x` 为该类的一个实例，则 `x[i]` 基本就等同于 `type(x).__getitem__(x, i)`。除非有说明例外情况，在没有定义适当方法的情况下尝试执行一种操作将引发一个异常 (通常为 [`AttributeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#AttributeError) 或 [`TypeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#TypeError))。

将一个特殊方法设为 `None` 表示对应的操作不可用。例如，如果一个类将 [`__iter__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__iter__) 设为 `None`，则该类就是不可迭代的，因此对其实例调用 [`iter()`](https://docs.python.org/zh-cn/3/library/functions.html#iter) 将引发一个 [`TypeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#TypeError) (而不会回退至 [`__getitem__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getitem__)). [2](https://docs.python.org/zh-cn/3/reference/datamodel.html#id9)

在实现模拟任何内置类型的类时，很重要的一点是模拟的实现程度对于被模拟对象来说应当是有意义的。例如，提取单个元素的操作对于某些序列来说是适宜的，但提取切片可能就没有意义。(这种情况的一个实例是 W3C 的文档对象模型中的 `NodeList` 接口。)



### 3.3.1. 基本定制

- `object.__new__`(*cls*[, *...*])

    调用以创建一个 *cls* 类的新实例。[`__new__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__new__) 是一个静态方法 (因为是特例所以你不需要显式地声明)，它会将所请求实例所属的类作为第一个参数。其余的参数会被传递给对象构造器表达式 (对类的调用)。[`__new__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__new__) 的返回值应为新对象实例 (通常是 *cls* 的实例)。
	
	典型的实现会附带适宜的参数使用 `super().__new__(cls[, ...])`，通过超类的 [`__new__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__new__) 方法来创建一个类的新实例，然后根据需要修改新创建的实例再将其返回。
	
	如果 [`__new__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__new__) 在构造对象期间被发起调用并且它返回了一个实例或 *cls* 的子类，则新实例的 [`__init__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__init__) 方法将以 `__init__(self[, ...])` 的形式被发起调用，其中 *self* 为新实例而其余的参数与被传给对象构造器的参数相同。如果 [`__new__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__new__) 未返回一个 *cls* 的实例，则新实例的 [`__init__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__init__) 方法就不会被执行。
	
	[`__new__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__new__) 的目的主要是允许不可变类型的子类 (例如 int, str 或 tuple) 定制实例创建过程。它也常会在自定义元类中被重载以便定制类创建过程。

- `object.__init__`(*self*[, *...*])

    在实例 (通过 [`__new__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__new__)) 被创建之后，返回调用者之前调用。其参数与传递给类构造器表达式的参数相同。一个基类如果有 [`__init__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__init__) 方法，则其所派生的类如果也有 [`__init__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__init__) 方法，就必须显式地调用它以确保实例基类部分的正确初始化；例如: `super().__init__([args...])`.
	
	因为对象是由 [`__new__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__new__) 和 [`__init__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__init__) 协作构造完成的 (由 [`__new__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__new__) 创建，并由 [`__init__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__init__) 定制)，所以 [`__init__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__init__) 返回的值只能是 `None`，否则会在运行时引发 [`TypeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#TypeError)。

- `object.__del__`(*self*)

    在实例将被销毁时调用。这还会调用终结器或析构器 (不适当)。如果一个基类具有 [`__del__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__del__) 方法，则其所派生的类如果也有 [`__del__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__del__) 方法，就必须显式地调用它以确保实例基类部分的正确清除。
	
	[`__del__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__del__) 方法可以 (但不推荐!) 通过创建一个该实例的新引用来推迟其销毁。这被称为对象 *重生*。[`__del__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__del__) 是否会在重生的对象将被销毁时再次被调用是由具体实现决定的 ；当前的 [CPython](https://docs.python.org/zh-cn/3/glossary.html#term-cpython) 实现只会调用一次。当解释器退出时不会确保为仍然存在的对象调用 [`__del__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__del__) 方法。
	
	> 注解 `del x` 并不直接调用 `x.__del__()` --- 前者会将 `x` 的引用计数减一，而后者仅会在 `x` 的引用计数变为零时被调用。
	
	**CPython implementation detail:** It is possible for a reference cycle to prevent the reference count of an object from going to zero. In this case, the cycle will be later detected and deleted by the [cyclic garbage collector](https://docs.python.org/zh-cn/3/glossary.html#term-garbage-collection). A common cause of reference cycles is when an exception has been caught in a local variable. The frame's locals then reference the exception, which references its own traceback, which references the locals of all frames caught in the traceback.
	
	参见 [`gc`](https://docs.python.org/zh-cn/3/library/gc.html#module-gc) 模块的文档。
	
	警告 由于调用 [`__del__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__del__) 方法时周边状况已不确定，在其执行期间发生的异常将被忽略，改为打印一个警告到 `sys.stderr`。特别地：
	- [`__del__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__del__) 可在任意代码被执行时启用，包括来自任意线程的代码。如果 [`__del__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__del__) 需要接受锁或启用其他阻塞资源，可能会发生死锁，例如该资源已被为执行 [`__del__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__del__) 而中断的代码所获取。
	- [`__del__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__del__) 可以在解释器关闭阶段被执行。因此，它需要访问的全局变量（包含其他模块）可能已被删除或设为 `None`。Python 会保证先删除模块中名称以单个下划线打头的全局变量再删除其他全局变量；如果已不存在其他对此类全局变量的引用，这有助于确保导入的模块在 [`__del__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__del__) 方法被调用时仍然可用。

- `object.__repr__`(*self*)

    由 [`repr()`](https://docs.python.org/zh-cn/3/library/functions.html#repr) 内置函数调用以输出一个对象的“官方”字符串表示。如果可能，这应类似一个有效的 Python 表达式，能被用来重建具有相同取值的对象（只要有适当的环境）。如果这不可能，则应返回形式如 `<...some useful description...>` 的字符串。返回值必须是一个字符串对象。如果一个类定义了 [`__repr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__repr__) 但未定义 [`__str__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__str__)，则在需要该类的实例的“非正式”字符串表示时也会使用 [`__repr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__repr__)。
	
	此方法通常被用于调试，因此确保其表示的内容包含丰富信息且无歧义是很重要的。

- `object.__str__`(*self*)

    通过 [`str(object)`](https://docs.python.org/zh-cn/3/library/stdtypes.html#str) 以及内置函数 [`format()`](https://docs.python.org/zh-cn/3/library/functions.html#format) 和 [`print()`](https://docs.python.org/zh-cn/3/library/functions.html#print) 调用以生成一个对象的“非正式”或格式良好的字符串表示。返回值必须为一个 [字符串](https://docs.python.org/zh-cn/3/library/stdtypes.html#textseq) 对象。
	
	此方法与 [`object.__repr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__repr__) 的不同点在于 [`__str__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__str__) 并不预期返回一个有效的 Python 表达式：可以使用更方便或更准确的描述信息。
	
	内置类型 [`object`](https://docs.python.org/zh-cn/3/library/functions.html#object) 所定义的默认实现会调用 [`object.__repr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__repr__)。

- `object.__bytes__`(*self*)

    通过 [bytes](https://docs.python.org/zh-cn/3/library/functions.html#func-bytes) 调用以生成一个对象的字节串表示。这应该返回一个 [`bytes`](https://docs.python.org/zh-cn/3/library/stdtypes.html#bytes) 对象。

- `object.__format__`(*self*, *format_spec*)

    通过 [`format()`](https://docs.python.org/zh-cn/3/library/functions.html#format) 内置函数、扩展、[格式化字符串字面值](https://docs.python.org/zh-cn/3/reference/lexical_analysis.html#f-strings) 的求值以及 [`str.format()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#str.format) 方法调用以生成一个对象的“格式化”字符串表示。 *format_spec* 参数为包含所需格式选项描述的字符串。 *format_spec* 参数的解读是由实现 [`__format__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__format__) 的类型决定的，不过大多数类或是将格式化委托给某个内置类型，或是使用相似的格式化选项语法。
	
	请参看 [格式规格迷你语言](https://docs.python.org/zh-cn/3/library/string.html#formatspec) 了解标准格式化语法的描述。返回值必须为一个字符串对象。
	
	*在 3.4 版更改:* `object` 本身的 __format__ 方法如果被传入任何非空字符，将会引发一个 [`TypeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#TypeError)。
	
	*在 3.7 版更改:* `object.__format__(x, '')` 现在等同于 `str(x)` 而不再是 `format(str(self), '')`。

- `object.__lt__`(*self*, *other*)

- `object.__le__`(*self*, *other*)

- `object.__eq__`(*self*, *other*)

- `object.__ne__`(*self*, *other*)

- `object.__gt__`(*self*, *other*)

- `object.__ge__`(*self*, *other*)

    以上这些被称为“富比较”方法。运算符号与方法名称的对应关系如下：`x<y` 调用 `x.__lt__(y)`、`x<=y` 调用 `x.__le__(y)`、`x==y` 调用 `x.__eq__(y)`、`x!=y` 调用 `x.__ne__(y)`、`x>y` 调用 `x.__gt__(y)`、`x>=y` 调用 `x.__ge__(y)`。
	
	如果指定的参数对没有相应的实现，富比较方法可能会返回单例对象 `NotImplemented`。按照惯例，成功的比较会返回 `False` 或 `True`。不过实际上这些方法可以返回任意值，因此如果比较运算符是要用于布尔值判断（例如作为 `if` 语句的条件），Python 会对返回值调用 [`bool()`](https://docs.python.org/zh-cn/3/library/functions.html#bool) 以确定结果为真还是假。
	
	在默认情况下 [`__ne__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__ne__) 会委托给 [`__eq__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__eq__) 并将结果取反，除非结果为 `NotImplemented`。比较运算符之间没有其他隐含关系，例如 `(x<y or x==y)` 为真并不意味着 `x<=y`。要根据单根运算自动生成排序操作，请参看 [`functools.total_ordering()`](https://docs.python.org/zh-cn/3/library/functools.html#functools.total_ordering)。
	
	请查看 [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__) 的相关段落，了解创建可支持自定义比较运算并可用作字典键的 [hashable](https://docs.python.org/zh-cn/3/glossary.html#term-hashable) 对象时要注意的一些事项。
	
	这些方法并没有对调参数版本（在左边参数不支持该操作但右边参数支持时使用）；而是 [`__lt__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__lt__) 和 [`__gt__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__gt__) 互为对方的反射， [`__le__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__le__) 和 [`__ge__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__ge__) 互为对方的反射，而 [`__eq__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__eq__) 和 [`__ne__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__ne__) 则是它们自己的反射。如果两个操作数的类型不同，且右操作数类型是左操作数类型的直接或间接子类，则优先选择右操作数的反射方法，否则优先选择左操作数的方法。虚拟子类不会被考虑。

- `object.__hash__`(*self*)

    通过内置函数 [`hash()`](https://docs.python.org/zh-cn/3/library/functions.html#hash) 调用以对哈希集的成员进行操作，属于哈希集的类型包括 [`set`](https://docs.python.org/zh-cn/3/library/stdtypes.html#set)、[`frozenset`](https://docs.python.org/zh-cn/3/library/stdtypes.html#frozenset) 以及 [`dict`](https://docs.python.org/zh-cn/3/library/stdtypes.html#dict)。[`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__) 应该返回一个整数。对象比较结果相同所需的唯一特征属性是其具有相同的哈希值；建议的做法是把参与比较的对象全部组件的哈希值混在一起，即将它们打包为一个元组并对该元组做哈希运算。例如:
	```python
	def __hash__(self):    
		return hash((self.name, self.nick, self.color)) 
	```
	
	>注解 [`hash()`](https://docs.python.org/zh-cn/3/library/functions.html#hash) 会从一个对象自定义的 [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__) 方法返回值中截断为 `Py_ssize_t` 的大小。通常对 64 位构建为 8 字节，对 32 位构建为 4 字节。如果一个对象的 [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__) 必须在不同位大小的构建上进行互操作，请确保检查全部所支持构建的宽度。做到这一点的简单方法是使用 `python -c "import sys; print(sys.hash_info.width)"`。
	
	
	如果一个类没有定义 [`__eq__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__eq__) 方法，那么也不应该定义 [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__) 操作；如果它定义了 [`__eq__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__eq__) 但没有定义 [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__)，则其实例将不可被用作可哈希集的项。如果一个类定义了可变对象并实现了 [`__eq__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__eq__) 方法，则不应该实现 [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__)，因为可哈希集的实现要求键的哈希集是不可变的（如果对象的哈希值发生改变，它将处于错误的哈希桶中）。
	
	用户定义的类默认带有 [`__eq__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__eq__) 和 [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__) 方法；使用它们与任何对象（自己除外）比较必定不相等，并且 `x.__hash__()` 会返回一个恰当的值以确保 `x == y` 同时意味着 `x is y` 且 `hash(x) == hash(y)`。
	
	一个类如果重载了 [`__eq__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__eq__) 且没有定义 [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__) 则会将其 [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__) 隐式地设为 `None`。当一个类的 [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__) 方法为 `None` 时，该类的实例将在一个程序尝试获取其哈希值时正确地引发 [`TypeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#TypeError)，并会在检测 `isinstance(obj, collections.abc.Hashable)` 时被正确地识别为不可哈希对象。
	
	如果一个重载了 [`__eq__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__eq__) 的类需要保留来自父类的 [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__) 实现，则必须通过设置 `__hash__ = <ParentClass>.__hash__` 来显式地告知解释器。
	
	如果一个没有重载 [`__eq__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__eq__) 的类需要去掉哈希支持，则应该在类定义中包含 `__hash__ = None`。一个自定义了 [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__) 以显式地引发 [`TypeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#TypeError) 的类会被 `isinstance(obj, collections.abc.Hashable)` 调用错误地识别为可哈希对象。
	
	> 注解:在默认情况下，str 和 bytes 对象的 [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__) 值会使用一个不可预知的随机值“加盐”。 虽然它们在一个单独 Python 进程中会保持不变，但它们的值在重复运行的 Python 间是不可预测的。这种做法是为了防止以下形式的拒绝服务攻击：通过仔细选择输入来利用字典插入操作在最坏情况下的执行效率即 O(n^2) 复杂度。详情见 http://www.ocert.org/advisories/ocert-2011-003.html改变哈希值会影响集合的迭代次序。Python 也从不保证这个次序不会被改变（通常它在 32 位和 64 位构建上是不一致的）。另见 [`PYTHONHASHSEED`](https://docs.python.org/zh-cn/3/using/cmdline.html#envvar-PYTHONHASHSEED).
	 
	*在 3.3 版更改:* 默认启用哈希随机化。

- `object.__bool__`(*self*)

    调用此方法以实现真值检测以及内置的 `bool()` 操作；应该返回 `False` 或 `True`。如果未定义此方法，则会查找并调用 [`__len__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__len__) 并在其返回非零值时视对象的逻辑值为真。如果一个类既未定义 [`__len__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__len__) 也未定义 [`__bool__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__bool__) 则视其所有实例的逻辑值为真。



### 3.3.2. 自定义属性访问

可以定义下列方法来自定义对类实例属性访问（`x.name` 的使用、赋值或删除）的具体含义.

- `object.__getattr__`(*self*, *name*)

    当默认属性访问因引发 [`AttributeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#AttributeError) 而失败时被调用 (可能是调用 [`__getattribute__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getattribute__) 时由于 *name* 不是一个实例属性或 `self` 的类关系树中的属性而引发了 [`AttributeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#AttributeError)；或者是对 *name* 特性属性调用 [`__get__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__get__) 时引发了 [`AttributeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#AttributeError))。此方法应当返回（找到的）属性值或是引发一个 [`AttributeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#AttributeError) 异常。
	
	请注意如果属性是通过正常机制找到的，[`__getattr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getattr__) 就不会被调用。（这是在 [`__getattr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getattr__) 和 [`__setattr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__setattr__) 之间故意设置的不对称性。）这既是出于效率理由也是因为不这样设置的话 [`__getattr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getattr__) 将无法访问实例的其他属性。要注意至少对于实例变量来说，你不必在实例属性字典中插入任何值（而是通过插入到其他对象）就可以模拟对它的完全控制。请参阅下面的 [`__getattribute__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getattribute__) 方法了解真正获取对属性访问的完全控制权的办法。

- `object.__getattribute__`(*self*, *name*)

    此方法会无条件地被调用以实现对类实例属性的访问。如果类还定义了 [`__getattr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getattr__)，则后者不会被调用，除非 [`__getattribute__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getattribute__) 显式地调用它或是引发了 [`AttributeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#AttributeError)。此方法应当返回（找到的）属性值或是引发一个 [`AttributeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#AttributeError) 异常。为了避免此方法中的无限递归，其实现应该总是调用具有相同名称的基类方法来访问它所需要的任何属性，例如 `object.__getattribute__(self, name)`。
	
	>注解: 此方法在作为通过特定语法或内置函数隐式地调用的结果的情况下查找特殊方法时仍可能会被跳过。参见 [特殊方法查找](https://docs.python.org/zh-cn/3/reference/datamodel.html#special-lookup)。

- `object.__setattr__`(*self*, *name*, *value*)

    此方法在一个属性被尝试赋值时被调用。这个调用会取代正常机制（即将值保存到实例字典）。 *name* 为属性名称， *value* 为要赋给属性的值。如果 [`__setattr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__setattr__) 想要赋值给一个实例属性，它应该调用同名的基类方法，例如 `object.__setattr__(self, name, value)`。

- `object.__delattr__`(*self*, *name*)

    类似于 [`__setattr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__setattr__) 但其作用为删除而非赋值。此方法应该仅在 `del obj.name` 对于该对象有意义时才被实现。

- `object.__dir__`(*self*)

    此方法会在对相应对象调用 [`dir()`](https://docs.python.org/zh-cn/3/library/functions.html#dir) 时被调用。返回值必须为一个序列。 [`dir()`](https://docs.python.org/zh-cn/3/library/functions.html#dir) 会把返回的序列转换为列表并对其排序。

#### 3.3.2.1. 自定义模块属性访问

特殊名称 `__getattr__` 和 `__dir__` 还可被用来自定义对模块属性的访问。模块层级的 `__getattr__` 函数应当接受一个参数，其名称为一个属性名，并返回计算结果值或引发一个 [`AttributeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#AttributeError)。如果通过正常查找即 [`object.__getattribute__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getattribute__) 未在模块对象中找到某个属性，则 `__getattr__` 会在模块的 `__dict__` 中查找，未找到时会引发一个 [`AttributeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#AttributeError)。如果找到，它会以属性名被调用并返回结果值。

`__dir__` 函数应当不接受任何参数，并且返回一个表示模块中可访问名称的字符串序列。 此函数如果存在，将会重载一个模块中的标准 [`dir()`](https://docs.python.org/zh-cn/3/library/functions.html#dir) 查找。

想要更细致地自定义模块的行为（设置属性和特性属性等待），可以将模块对象的 `__class__` 属性设置为一个 [`types.ModuleType`](https://docs.python.org/zh-cn/3/library/types.html#types.ModuleType) 的子类。例如:

```python
import sys
from types import ModuleType

class VerboseModule(ModuleType):
    def __repr__(self):
        return f'Verbose {self.__name__}'

    def __setattr__(self, attr, value):
        print(f'Setting {attr}...')
        super().__setattr__(attr, value)

sys.modules[__name__].__class__ = VerboseModule
```

> **注解:** 定义模块的 `__getattr__` 和设置模块的 `__class__` 只会影响使用属性访问语法进行的查找 -- 直接访问模块全局变量（不论是通过模块内的代码还是通过对模块全局字典的引用）是不受影响的。

*在 3.5 版更改:* `__class__` 模块属性改为可写。

*3.7 新版功能:* `__getattr__` 和 `__dir__` 模块属性。

参见
- [**PEP 562**](https://www.python.org/dev/peps/pep-0562) - 模块 __getattr__ 和 __dir__

    描述用于模块的 `__getattr__` 和 `__dir__` 函数。



#### 3.3.2.2. 实现描述器

以下方法仅当一个包含该方法的类（称为 *描述器* 类）的实例出现于一个 *所有者* 类中的时候才会起作用（该描述器必须在所有者类或其某个上级类的字典中）。在以下示例中，“属性”指的是名称为所有者类 [`__dict__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#object.__dict__) 中的特征属性的键名的属性。

- `object.__get__`(*self*, *instance*, *owner=None*)

    调用此方法以获取所有者类的属性（类属性访问）或该类的实例的属性（实例属性访问）。 可选的 *owner* 参数是所有者类而 *instance* 是被用来访问属性的实例，如果通过 *owner* 来访问属性则返回 `None`。此方法应当返回计算得到的属性值或是引发 [`AttributeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#AttributeError) 异常。
	
	[**PEP 252**](https://www.python.org/dev/peps/pep-0252) 指明 [`__get__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__get__) 为带有一至二个参数的可调用对象。 Python 自身内置的描述器支持此规格定义；但是，某些第三方工具可能要求必须带两个参数。 Python 自身的 [`__getattribute__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getattribute__) 实现总是会传入两个参数，无论它们是否被要求提供。

- `object.__set__`(*self*, *instance*, *value*)

    调用此方法以设置 *instance* 指定的所有者类的实例的属性为新值 *value*。
	
	请注意，添加 [`__set__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__set__) 或 [`__delete__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__delete__) 会将描述器变成“数据描述器”。 更多细节请参阅 [调用描述器](https://docs.python.org/zh-cn/3/reference/datamodel.html#descriptor-invocation)。

- `object.__delete__`(*self*, *instance*)

    调用此方法以删除 *instance* 指定的所有者类的实例的属性。

- `object.__set_name__`(*self*, *owner*, *name*)

    在所有者类 *owner* 创建时被调用。描述器会被赋值给 *name*。
	>**注解:** [`__set_name__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__set_name__) 只是作为 [`type`](https://docs.python.org/zh-cn/3/library/functions.html#type) 构造器的一部分被隐式地调用，因此在某个类被初次创建之后又额外添加一个描述器时，那就需要显式地调用它并且附带适当的形参:
	```python
	class A:   
		pass 
	descr = custom_descriptor() 
	A.attr = descr 
	descr.__set_name__(A, 'attr') `
	```
	> 详情参见 [创建类对象](https://docs.python.org/zh-cn/3/reference/datamodel.html#class-object-creation)。*3.6 新版功能.*

属性 `__objclass__` 会被 [`inspect`](https://docs.python.org/zh-cn/3/library/inspect.html#module-inspect) 模块解读为指定此对象定义所在的类（正确设置此属性有助于动态类属性的运行时内省）。对于可调用对象来说，它可以指明预期或要求提供一个特定类型（或子类）的实例作为第一个位置参数（例如，CPython 会为实现于 C 中的未绑定方法设置此属性）。



#### 3.3.2.3. 调用描述器

总的说来，描述器就是具有“绑定行为”的对象属性，其属性访问已被描述器协议中的方法所重载，包括 [`__get__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__get__), [`__set__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__set__) 和 [`__delete__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__delete__)。如果一个对象定义了以上方法中的任意一个，它就被称为描述器。

属性访问的默认行为是从一个对象的字典中获取、设置或删除属性。例如，`a.x` 的查找顺序会从 `a.__dict__['x']` 开始，然后是 `type(a).__dict__['x']`，接下来依次查找 `type(a)` 的上级基类，不包括元类。

但是，如果找到的值是定义了某个描述器方法的对象，则 Python 可能会重载默认行为并转而发起调用描述器方法。这具体发生在优先级链的哪个环节则要根据所定义的描述器方法及其被调用的方式来决定。

描述器发起调用的开始点是一个绑定 `a.x`。参数的组合方式依 `a` 而定:

- 直接调用

    最简单但最不常见的调用方式是用户代码直接发起调用一个描述器方法: `x.__get__(a)`。

- 实例绑定

    如果绑定到一个对象实例，`a.x` 会被转换为调用: `type(a).__dict__['x'].__get__(a, type(a))`。

- 类绑定

    如果绑定到一个类，`A.x` 会被转换为调用: `A.__dict__['x'].__get__(None, A)`。

- 超绑定

    如果 `a` 是 [`super`](https://docs.python.org/zh-cn/3/library/functions.html#super) 的一个实例，则绑定 `super(B, obj).m()` 会在 `obj.__class__.__mro__` 中搜索 `B` 的直接上级基类 `A` 然后通过以下调用发起调用描述器: `A.__dict__['m'].__get__(obj, obj.__class__)`。

对于实例绑定，发起描述器调用的优先级取决于定义了哪些描述器方法。一个描述器可以定义 [`__get__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__get__)、[`__set__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__set__) 和 [`__delete__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__delete__) 的任意组合。如果它没有定义 [`__get__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__get__)，则访问属性会返回描述器对象自身，除非对象的实例字典中有相应属性值。如果描述器定义了 [`__set__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__set__) 和/或 [`__delete__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__delete__)，则它是一个数据描述器；如果以上两个都未定义，则它是一个非数据描述器。通常，数据描述器会同时定义 [`__get__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__get__) 和 [`__set__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__set__)，而非数据描述器只有 [`__get__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__get__) 方法。定义了 [`__set__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__set__) 和 [`__get__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__get__) 的数据描述器总是会重载实例字典中的定义。与之相对的，非数据描述器可被实例所重载。

Python 方法 (包括 [`staticmethod()`](https://docs.python.org/zh-cn/3/library/functions.html#staticmethod) 和 [`classmethod()`](https://docs.python.org/zh-cn/3/library/functions.html#classmethod)) 都是作为非描述器来实现的。因此实例可以重定义并重载方法。这允许单个实例获得与相同类的其他实例不一样的行为。

[`property()`](https://docs.python.org/zh-cn/3/library/functions.html#property) 函数是作为数据描述器来实现的。因此实例不能重载特性属性的行为。



#### 3.3.2.4. __slots__

*__slots__* 允许我们显式地声明数据成员（例如特征属性）并禁止创建 *__dict__* 和 *__weakref__* (除非是在 *__slots__* 中显式地声明或是在父类中可用。)

相比使用 *__dict__* 此方式可以显著地节省空间。 属性查找速度也可得到显著的提升。

- `object.__slots__`

    这个类变量可赋值为字符串、可迭代对象或由实例使用的变量名构成的字符串序列。 *__slots__* 会为已声明的变量保留空间，并阻止自动为每个实例创建 *__dict__* 和 *__weakref__*。

##### 3.3.2.4.1. 使用 *__slots__* 的注意事项

- 当继承自一个未定义 *__slots__* 的类时，实例的 *__dict__* 和 *__weakref__* 属性将总是可访问。
- 没有 *__dict__* 变量，实例就不能给未在 *__slots__* 定义中列出的新变量赋值。尝试给一个未列出的变量名赋值将引发 [`AttributeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#AttributeError)。新变量需要动态赋值，就要将 `'__dict__'` 加入到 *__slots__* 声明的字符串序列中。
- 如果未给每个实例设置 *__weakref__* 变量，定义了 *__slots__* 的类就不支持对其实际的弱引用。如果需要弱引用支持，就要将 `'__weakref__'` 加入到 *__slots__* 声明的字符串序列中。
- *__slots__* 是通过为每个变量名创建描述器 ([实现描述器](https://docs.python.org/zh-cn/3/reference/datamodel.html#descriptors)) 在类层级上实现的。因此，类属性不能被用来为通过 *__slots__* 定义的实例变量设置默认值；否则，类属性就会覆盖描述器赋值。
- *__slots__* 声明的作用不只限于定义它的类。在父类中声明的 *__slots__* 在其子类中同样可用。不过，子类将会获得 *__dict__* 和 *__weakref__* 除非它们也定义了 *__slots__* (其中应该仅包含对任何 *额外* 名称的声明位置)。
- 如果一个类定义的位置在某个基类中也有定义，则由基类位置定义的实例变量将不可访问（除非通过直接从基类获取其描述器的方式）。这会使得程序的含义变成未定义。未来可能会添加一个防止此情况的检查。
- 非空的 *__slots__* 不适用于派生自“可变长度”内置类型例如 [`int`](https://docs.python.org/zh-cn/3/library/functions.html#int)、[`bytes`](https://docs.python.org/zh-cn/3/library/stdtypes.html#bytes) 和 [`tuple`](https://docs.python.org/zh-cn/3/library/stdtypes.html#tuple) 的派生类。
- 任何非字符串可迭代对象都可以被赋值给 *__slots__*。映射也可以被使用；不过，未来可能会分别赋给每个键具有特殊含义的值。
- *__class__* 赋值仅在两个类具有相同的 *__slots__* 时才会起作用。
- 带有多个父类声明位置的多重继承也是可用的，但仅允许一个父类具有由声明位置创建的属性（其他基类必须具有空的位置布局） —— 违反规则将引发 [`TypeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#TypeError)。
- 如果为 *__slots__* 使用了一个迭代器，则会为迭代器的每个值创建描述器。 但是 *__slots__* 属性将为一个空迭代器。



### 3.3.3. 自定义类创建

当一个类继承其他类时，那个类的 *__init_subclass__* 会被调用。这样就可以编写能够改变子类行为的类。这与类装饰器有紧密的关联，但是类装饰器是影响它们所应用的特定类，而 `__init_subclass__` 则只作用于定义了该方法的类所派生的子类。

- *classmethod* `object.``__init_subclass__`(*cls*)

    当所在类派生子类时此方法就会被调用。*cls* 将指向新的子类。如果定义为一个普通实例方法，此方法将被隐式地转换为类方法。
	
	传入一个新类的关键字参数会被传给父类的 `__init_subclass__`。为了与其他使用 `__init_subclass__` 的类兼容，应当根据需要去掉部分关键字参数再将其余的传给基类，例如:
	```python
	class Philosopher:    
		def __init_subclass__(cls, /, default_name, **kwargs):
			super().__init_subclass__(**kwargs)      
			cls.default_name = default_name 
	class AustralianPhilosopher(Philosopher, default_name="Bruce"):    	
		pass 
	```
	`object.__init_subclass__` 的默认实现什么都不做，只在带任意参数调用时引发一个错误。
	> 注解 元类提示 `metaclass` 将被其它类型机制消耗掉，并不会被传给 `__init_subclass__` 的实现。实际的元类（而非显式的提示）可通过 `type(cls)` 访问。
	
	*3.6 新版功能.*



#### 3.3.3.1. 元类

默认情况下，类是使用 [`type()`](https://docs.python.org/zh-cn/3/library/functions.html#type) 来构建的。类体会在一个新的命名空间内执行，类名会被局部绑定到 `type(name, bases, namespace)` 的结果。

类创建过程可通过在定义行传入 `metaclass` 关键字参数，或是通过继承一个包含此参数的现有类来进行定制。在以下示例中，`MyClass` 和 `MySubclass` 都是 `Meta` 的实例:

```python
class Meta(type):
    pass

class MyClass(metaclass=Meta):
    pass

class MySubclass(MyClass):
    pass
```

在类定义内指定的任何其他关键字参数都会在下面所描述的所有元类操作中进行传递。

当一个类定义被执行时，将发生以下步骤:

- 解析 MRO 条目；
- 确定适当的元类；
- 准备类命名空间；
- 执行类主体；
- 创建类对象。

#### 3.3.3.2. 解析 MRO 条目

如果在类定义中出现的基类不是 [`type`](https://docs.python.org/zh-cn/3/library/functions.html#type) 的实例，则使用 `__mro_entries__` 方法对其进行搜索，当找到结果时，它会以原始基类元组做参数进行调用。此方法必须返回类的元组以替代此基类被使用。元组可以为空，在此情况下原始基类将被忽略。

参见
[**PEP 560**](https://www.python.org/dev/peps/pep-0560) - 对 typing 模块和泛型类型的核心支持

#### 3.3.3.3. 确定适当的元类

为一个类定义确定适当的元类是根据以下规则:

- 如果没有基类且没有显式指定元类，则使用 [`type()`](https://docs.python.org/zh-cn/3/library/functions.html#type)；
- 如果给出一个显式元类而且 *不是* [`type()`](https://docs.python.org/zh-cn/3/library/functions.html#type) 的实例，则其会被直接用作元类；
- 如果给出一个 [`type()`](https://docs.python.org/zh-cn/3/library/functions.html#type) 的实例作为显式元类，或是定义了基类，则使用最近派生的元类。

最近派生的元类会从显式指定的元类（如果有）以及所有指定的基类的元类（即 `type(cls)`）中选取。最近派生的元类应为 *所有* 这些候选元类的一个子类型。如果没有一个候选元类符合该条件，则类定义将失败并抛出 `TypeError`。



#### 3.3.3.4. 准备类命名空间

一旦确定了适当的元类，则将准备好类命名空间。 如果元类具有 `__prepare__` 属性，它会以 `namespace = metaclass.__prepare__(name, bases, **kwds)` 的形式被调用（其中如果有任何额外的关键字参数，则应当来自类定义）。 `__prepare__` 方法应该被实现为 [`classmethod()`](https://docs.python.org/zh-cn/3/library/functions.html#classmethod)。 `__prepare__` 所返回的命名空间会被传入 `__new__`，但是当最终的类对象被创建时，该命名空间会被拷贝到一个新的 `dict` 中。

如果元类没有 `__prepare__` 属性，则类命名空间将初始化为一个空的有序映射。

参见
- [**PEP 3115**](https://www.python.org/dev/peps/pep-3115) - Python 3000 中的元类
    引入 `__prepare__` 命名空间钩子

#### 3.3.3.5. 执行类主体

类主体会以（类似于） `exec(body, globals(), namespace)` 的形式被执行。普通调用与 [`exec()`](https://docs.python.org/zh-cn/3/library/functions.html#exec) 的关键区别在于当类定义发生于函数内部时，词法作用域允许类主体（包括任何方法）引用来自当前和外部作用域的名称。

但是，即使当类定义发生于函数内部时，在类内部定义的方法仍然无法看到在类作用域层次上定义的名称。类变量必须通过实例的第一个形参或类方法来访问，或者是通过下一节中描述的隐式词法作用域的 `__class__` 引用。



#### 3.3.3.6. 创建类对象

一旦执行类主体完成填充类命名空间，将通过调用 `metaclass(name, bases, namespace, **kwds)` 创建类对象（此处的附加关键字参数与传入 `__prepare__` 的相同）。

如果类主体中有任何方法引用了 `__class__` 或 `super`，这个类对象会通过零参数形式的 [`super()`](https://docs.python.org/zh-cn/3/library/functions.html#super). `__class__` 所引用，这是由编译器所创建的隐式闭包引用。这使用零参数形式的 [`super()`](https://docs.python.org/zh-cn/3/library/functions.html#super) 能够正确标识正在基于词法作用域来定义的类，而被用于进行当前调用的类或实例则是基于传递给方法的第一个参数来标识的。

**CPython implementation detail:** 在 CPython 3.6 及之后的版本中，`__class__` 单元会作为类命名空间中的 `__classcell__` 条目被传给元类。 如果存在，它必须被向上传播给 `type.__new__` 调用，以便能正确地初始化该类。 如果不这样做，在 Python 3.8 中将引发 [`RuntimeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#RuntimeError)。

当使用默认的元类 [`type`](https://docs.python.org/zh-cn/3/library/functions.html#type) 或者任何最终会调用 `type.__new__` 的元类时，以下额外的自定义步骤将在创建类对象之后被发起调用:

- 首先，`type.__new__` 将收集类命名空间中所有定义了 [`__set_name__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__set_name__) 方法的描述器；
- 接下来，所有这些 `__set_name__` 方法将使用所定义的类和特定描述器所赋的名称进行调用；
- 最后，将在新类根据方法解析顺序所确定的直接父类上调用 [`__init_subclass__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__init_subclass__) 钩子。

在类对象创建之后，它会被传给包含在类定义中的类装饰器（如果有的话），得到的对象将作为已定义的类绑定到局部命名空间。

当通过 `type.__new__` 创建一个新类时，提供以作为命名空间形参的对象会被复制到一个新的有序映射并丢弃原对象。这个新副本包装于一个只读代理中，后者则成为类对象的 [`__dict__`](https://docs.python.org/zh-cn/3/library/stdtypes.html#object.__dict__) 属性。

参见

- [**PEP 3135**](https://www.python.org/dev/peps/pep-3135) - 新的超类型

    描述隐式的 `__class__` 闭包引用

#### 3.3.3.7. 元类的作用

元类的潜在作用非常广泛。已经过尝试的设想包括枚举、日志、接口检查、自动委托、自动特征属性创建、代理、框架以及自动资源锁定/同步等等。

### 3.3.4. 自定义实例及子类检查

以下方法被用来重载 [`isinstance()`](https://docs.python.org/zh-cn/3/library/functions.html#isinstance) 和 [`issubclass()`](https://docs.python.org/zh-cn/3/library/functions.html#issubclass) 内置函数的默认行为。

特别地，元类 [`abc.ABCMeta`](https://docs.python.org/zh-cn/3/library/abc.html#abc.ABCMeta) 实现了这些方法以便允许将抽象基类（ABC）作为“虚拟基类”添加到任何类或类型（包括内置类型），包括其他 ABC 之中。

- `class.__instancecheck__`(*self*, *instance*)

    如果 *instance* 应被视为 *class* 的一个（直接或间接）实例则返回真值。如果定义了此方法，则会被调用以实现 `isinstance(instance, class)`。

- `class.__subclasscheck__`(*self*, *subclass*)

    Return true 如果 *subclass* 应被视为 *class* 的一个（直接或间接）子类则返回真值。如果定义了此方法，则会被调用以实现 `issubclass(subclass, class)`。

请注意这些方法的查找是基于类的类型（元类）。它们不能作为类方法在实际的类中被定义。这与基于实例被调用的特殊方法的查找是一致的，只有在此情况下实例本身被当作是类。

参见
- [**PEP 3119**](https://www.python.org/dev/peps/pep-3119) - 引入抽象基类

    新增功能描述，通过 [`__instancecheck__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#class.__instancecheck__) 和 [`__subclasscheck__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#class.__subclasscheck__) 来定制 [`isinstance()`](https://docs.python.org/zh-cn/3/library/functions.html#isinstance) 和 [`issubclass()`](https://docs.python.org/zh-cn/3/library/functions.html#issubclass) 行为，加入此功能的动机是出于向该语言添加抽象基类的内容（参见 [`abc`](https://docs.python.org/zh-cn/3/library/abc.html#module-abc) 模块）。

### 3.3.5. 模拟泛型类型

通过定义一个特殊方法，可以实现由 [**PEP 484**](https://www.python.org/dev/peps/pep-0484) 所规定的泛型类语法 (例如 `List[int]`):

- *classmethod* `object.__class_getitem__`(*cls*, *key*)

    按照 *key* 参数指定的类型返回一个表示泛型类的专门化对象。

此方法的查找会基于对象自身，并且当定义于类体内部时，此方法将隐式地成为类方法。请注意，此机制主要是被保留用于静态类型提示，不鼓励在其他场合使用。

参见
[**PEP 560**](https://www.python.org/dev/peps/pep-0560) - 对 typing 模块和泛型类型的核心支持



### 3.3.6. 模拟可调用对象

- `object.__call__`(*self*[, *args...*])

    此方法会在实例作为一个函数被“调用”时被调用；如果定义了此方法，则 `x(arg1, arg2, ...)` 就相当于 `x.__call__(arg1, arg2, ...)` 的快捷方式。



### 3.3.7. 模拟容器类型

可以定义下列方法来实现容器对象。 容器通常属于序列（如列表或元组）或映射（如字典），但也存在其他形式的容器。 前几个方法集被用于模拟序列或是模拟映射；两者的不同之处在于序列允许的键应为整数 *k* 且 `0 <= k < N` 其中 *N* 是序列或定义指定区间的项的切片对象的长度。 此外还建议让映射提供 `keys()`, `values()`, `items()`, `get()`, `clear()`, `setdefault()`, `pop()`, `popitem()`, `copy()` 以及 `update()` 等方法，它们的行为应与 Python 标准字典对象的相应方法类似。 此外 [`collections.abc`](https://docs.python.org/zh-cn/3/library/collections.abc.html#module-collections.abc) 模块提供了一个 [`MutableMapping`](https://docs.python.org/zh-cn/3/library/collections.abc.html#collections.abc.MutableMapping) 抽象基类以便根据由 [`__getitem__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getitem__), [`__setitem__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__setitem__), [`__delitem__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__delitem__), 和 `keys()` 组成的基本集来创建所需的方法。 可变序列还应像 Python 标准列表对象那样提供 `append()`, `count()`, `index()`, `extend()`, `insert()`, `pop()`, `remove()`, `reverse()` 和 `sort()` 等方法。 最后，序列类型还应通过定义下文描述的 [`__add__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__add__), [`__radd__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__radd__), [`__iadd__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__iadd__), [`__mul__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__mul__), [`__rmul__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__rmul__) 和 [`__imul__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__imul__) 等方法来实现加法（指拼接）和乘法（指重复）；它们不应定义其他数值运算符。 此外还建议映射和序列都实现 [`__contains__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__contains__) 方法以允许高效地使用 `in` 运算符；对于映射，`in` 应该搜索映射的键；对于序列，则应搜索其中的值。 另外还建议映射和序列都实现 [`__iter__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__iter__) 方法以允许高效地迭代容器中的条目；对于映射，[`__iter__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__iter__) 应当迭代对象的键；对于序列，则应当迭代其中的值。

- `object.__len__`(*self*)

    调用此方法以实现内置函数 [`len()`](https://docs.python.org/zh-cn/3/library/functions.html#len)。应该返回对象的长度，以一个 `>=` 0 的整数表示。此外，如果一个对象未定义 [`__bool__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__bool__) 方法而其 [`__len__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__len__) 方法返回值为零，则在布尔运算中会被视为假值。
	
	**CPython implementation detail:** 在 CPython 中，要求长度最大为 [`sys.maxsize`](https://docs.python.org/zh-cn/3/library/sys.html#sys.maxsize)。如果长度大于 `sys.maxsize` 则某些特性 (例如 [`len()`](https://docs.python.org/zh-cn/3/library/functions.html#len)) 可能会引发 [`OverflowError`](https://docs.python.org/zh-cn/3/library/exceptions.html#OverflowError)。要通过真值检测来防止引发 `OverflowError`，对象必须定义 [`__bool__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__bool__) 方法。

- `object.__length_hint__`(*self*)

    调用此方法以实现 [`operator.length_hint()`](https://docs.python.org/zh-cn/3/library/operator.html#operator.length_hint)。 应该返回对象长度的估计值（可能大于或小于实际长度）。 此长度应为一个 `>=` 0 的整数。 返回值也可以为 [`NotImplemented`](https://docs.python.org/zh-cn/3/library/constants.html#NotImplemented)，这会被视作与 `__length_hint__` 方法完全不存在时一样处理。 此方法纯粹是为了优化性能，并不要求正确无误。
	
	*3.4 新版功能.*

	>注解
	切片是通过下述三个专门方法完成的。以下形式的调用
	```python
	a[1:2] = b
	```

	会为转写为

	```python
	a[slice(1, 2, None)] = b
	```

	其他形式以此类推。略去的切片项总是以 `None` 补全。

- `object.__getitem__`(*self*, *key*)

    调用此方法以实现 `self[key]` 的求值。对于序列类型，接受的键应为整数和切片对象。请注意负数索引（如果类想要模拟序列类型）的特殊解读是取决于 [`__getitem__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getitem__) 方法。如果 *key* 的类型不正确则会引发 [`TypeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#TypeError) 异常；如果为序列索引集范围以外的值（在进行任何负数索引的特殊解读之后）则应引发 [`IndexError`](https://docs.python.org/zh-cn/3/library/exceptions.html#IndexError) 异常。对于映射类型，如果 *key* 找不到（不在容器中）则应引发 [`KeyError`](https://docs.python.org/zh-cn/3/library/exceptions.html#KeyError) 异常。
	
	> **注解:**[`for`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#for) 循环在有不合法索引时会期待捕获 [`IndexError`](https://docs.python.org/zh-cn/3/library/exceptions.html#IndexError) 以便正确地检测到序列的结束。

- `object.__setitem__`(*self*, *key*, *value*)

    调用此方法以实现向 `self[key]` 赋值。注意事项与 [`__getitem__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getitem__) 相同。为对象实现此方法应该仅限于需要映射允许基于键修改值或添加键，或是序列允许元素被替换时。不正确的 *key* 值所引发的异常应与 [`__getitem__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getitem__) 方法的情况相同。

- `object.__delitem__`(*self*, *key*)

    调用此方法以实现 `self[key]` 的删除。注意事项与 [`__getitem__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getitem__) 相同。为对象实现此方法应该权限于需要映射允许移除键，或是序列允许移除元素时。不正确的 *key* 值所引发的异常应与 [`__getitem__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getitem__) 方法的情况相同。

- `object.__missing__`(*self*, *key*)

    此方法由 [`dict`](https://docs.python.org/zh-cn/3/library/stdtypes.html#dict).[`__getitem__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getitem__) 在找不到字典中的键时调用以实现 dict 子类的 `self[key]`。

- `object.__iter__`(*self*)

    此方法在需要为容器创建迭代器时被调用。此方法应该返回一个新的迭代器对象，它能够逐个迭代容器中的所有对象。对于映射，它应该逐个迭代容器中的键。迭代器对象也需要实现此方法；它们需要返回对象自身。有关迭代器对象的详情请参看 [迭代器类型](https://docs.python.org/zh-cn/3/library/stdtypes.html#typeiter) 一节。

- `object.__reversed__`(*self*)

    此方法（如果存在）会被 [`reversed()`](https://docs.python.org/zh-cn/3/library/functions.html#reversed) 内置函数调用以实现逆向迭代。它应当返回一个新的以逆序逐个迭代容器内所有对象的迭代器对象。如果未提供 [`__reversed__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__reversed__) 方法，则 [`reversed()`](https://docs.python.org/zh-cn/3/library/functions.html#reversed) 内置函数将回退到使用序列协议 ([`__len__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__len__) 和 [`__getitem__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getitem__))。支持序列协议的对象应当仅在能够提供比 [`reversed()`](https://docs.python.org/zh-cn/3/library/functions.html#reversed) 所提供的实现更高效的实现时才提供 [`__reversed__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__reversed__) 方法。

成员检测运算符 ([`in`](https://docs.python.org/zh-cn/3/reference/expressions.html#in) 和 [`not in`](https://docs.python.org/zh-cn/3/reference/expressions.html#not-in)) 通常以对容器进行逐个迭代的方式来实现。 不过，容器对象可以提供以下特殊方法并采用更有效率的实现，这样也不要求对象必须为可迭代对象。

- `object.__contains__`(*self*, *item*)

    调用此方法以实现成员检测运算符。如果 *item* 是 *self* 的成员则应返回真，否则返回假。对于映射类型，此检测应基于映射的键而不是值或者键值对。对于未定义 [`__contains__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__contains__) 的对象，成员检测将首先尝试通过 [`__iter__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__iter__) 进行迭代，然后再使用 [`__getitem__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getitem__) 的旧式序列迭代协议，参看 [语言参考中的相应部分](https://docs.python.org/zh-cn/3/reference/expressions.html#membership-test-details)。



### 3.3.8. 模拟数字类型

定义以下方法即可模拟数字类型。特定种类的数字不支持的运算（例如非整数不能进行位运算）所对应的方法应当保持未定义状态。

- `object.__add__`(*self*, *other*)

- `object.__sub__`(*self*, *other*)

- `object.__mul__`(*self*, *other*)

- `object.__matmul__`(*self*, *other*)

- `object.__truediv__`(*self*, *other*)

- `object.__floordiv__`(*self*, *other*)

- `object.__mod__`(*self*, *other*)

- `object.__divmod__`(*self*, *other*)

- `object.__pow__`(*self*, *other*[, *modulo*])

- `object.__lshift__`(*self*, *other*)

- `object.__rshift__`(*self*, *other*)

- `object.__and__`(*self*, *other*)

- `object.__xor__`(*self*, *other*)

- `object.__or__`(*self*, *other*)

    调用这些方法来实现二进制算术运算 (`+`, `-`, `*`, `@`, `/`, `//`, `%`, [`divmod()`](https://docs.python.org/zh-cn/3/library/functions.html#divmod), [`pow()`](https://docs.python.org/zh-cn/3/library/functions.html#pow), `**`, `<<`, `>>`, `&`, `^`, `|`)。例如，求表达式 `x + y` 的值，其中 *x* 是具有 [`__add__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__add__) 方法的类的一个实例，则会调用 `x.__add__(y)`。[`__divmod__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__divmod__) 方法应该等价于使用 [`__floordiv__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__floordiv__) 和 [`__mod__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__mod__)，它不应该被关联到 [`__truediv__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__truediv__)。请注意如果要支持三元版本的内置 [`pow()`](https://docs.python.org/zh-cn/3/library/functions.html#pow) 函数，则 [`__pow__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__pow__) 的定义应该接受可选的第三个参数。
	
	如果这些方法中的某一个不支持与所提供参数进行运算，它应该返回 `NotImplemented`。

- `object.__radd__`(*self*, *other*)

- `object.__rsub__`(*self*, *other*)

- `object.__rmul__`(*self*, *other*)

- `object.__rmatmul__`(*self*, *other*)

- `object.__rtruediv__`(*self*, *other*)

- `object.__rfloordiv__`(*self*, *other*)

- `object.__rmod__`(*self*, *other*)

- `object.__rdivmod__`(*self*, *other*)

- `object.__rpow__`(*self*, *other*[, *modulo*])

- `object.__rlshift__`(*self*, *other*)

- `object.__rrshift__`(*self*, *other*)

- `object.__rand__`(*self*, *other*)

- `object.__rxor__`(*self*, *other*)

- `object.__ror__`(*self*, *other*)

    调用这些方法来实现具有反射（交换）操作数的二进制算术运算 (`+`, `-`, `*`, `@`, `/`, `//`, `%`, [`divmod()`](https://docs.python.org/zh-cn/3/library/functions.html#divmod), [`pow()`](https://docs.python.org/zh-cn/3/library/functions.html#pow), `**`, `<<`, `>>`, `&`, `^`, `|`)。这些成员函数仅会在左操作数不支持相应运算 [3](https://docs.python.org/zh-cn/3/reference/datamodel.html#id10) 且两个操作数类型不同时被调用。[4](https://docs.python.org/zh-cn/3/reference/datamodel.html#id11) 例如，求表达式 `x - y` 的值，其中 *y* 是具有 [`__rsub__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__rsub__) 方法的类的一个实例，则当 `x.__sub__(y)` 返回 *NotImplemented* 时会调用 `y.__rsub__(x)`。
	
	请注意三元版的 [`pow()`](https://docs.python.org/zh-cn/3/library/functions.html#pow) 并不会尝试调用 [`__rpow__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__rpow__) (因为强制转换规则会太过复杂)。
	
	>**注解:** 如果右操作数类型为左操作数类型的一个子类，且该子类提供了指定运算的反射方法，则此方法会先于左操作数的非反射方法被调用。此行为可允许子类重载其祖先类的运算符。

- `object.__iadd__`(*self*, *other*)

- `object.__isub__`(*self*, *other*)

- `object.__imul__`(*self*, *other*)

- `object.__imatmul__`(*self*, *other*)

- `object.__itruediv__`(*self*, *other*)

- `object.__ifloordiv__`(*self*, *other*)

- `object.__imod__`(*self*, *other*)

- `object.__ipow__`(*self*, *other*[, *modulo*])

- `object.__ilshift__`(*self*, *other*)

- `object.__irshift__`(*self*, *other*)

- `object.__iand__`(*self*, *other*)

- `object.__ixor__`(*self*, *other*)

- `object.``__ior__`(*self*, *other*)

    调用这些方法来实现扩展算术赋值 (`+=`, `-=`, `*=`, `@=`, `/=`, `//=`, `%=`, `**=`, `<<=`, `>>=`, `&=`, `^=`, `|=`)。这些方法应该尝试进行自身操作 (修改 *self*) 并返回结果 (结果应该但并非必须为 *self*)。如果某个方法未被定义，相应的扩展算术赋值将回退到普通方法。例如，如果 *x* 是具有 [`__iadd__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__iadd__) 方法的类的一个实例，则 `x += y` 就等价于 `x = x.__iadd__(y)`。否则就如 `x + y` 的求值一样选择 `x.__add__(y)` 和 `y.__radd__(x)`。在某些情况下，扩展赋值可导致未预期的错误 (参见 [为什么 a_tuple[i\] += ['item'] 会在执行加法时引发异常？](https://docs.python.org/zh-cn/3/faq/programming.html#faq-augmented-assignment-tuple-error))，但此行为实际上是数据模型的一个组成部分。
	
	>**注解:** 由于 `**=` 的分配机制存在一个错误，定义了 [`__ipow__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__ipow__) 但是返回 `NotImplemented` 的类将无法回退到 `x.__pow__(y)` 和 `y.__rpow__(x)`。 此错误在 Python 3.10 中被修正。

- `object.__neg__`(*self*)

- `object.__pos__`(*self*)

- `object.__abs__`(*self*)

- `object.__invert__`(*self*)

    调用此方法以实现一元算术运算 (`-`, `+`, [`abs()`](https://docs.python.org/zh-cn/3/library/functions.html#abs) 和 `~`)。

- `object.__complex__`(*self*)

- `object.__int__`(*self*)

- `object.__float__`(*self*)

    调用这些方法以实现内置函数 [`complex()`](https://docs.python.org/zh-cn/3/library/functions.html#complex), [`int()`](https://docs.python.org/zh-cn/3/library/functions.html#int) 和 [`float()`](https://docs.python.org/zh-cn/3/library/functions.html#float)。应当返回一个相应类型的值。

- `object.__index__`(*self*)

    调用此方法以实现 [`operator.index()`](https://docs.python.org/zh-cn/3/library/operator.html#operator.index) 以及 Python 需要无损地将数字对象转换为整数对象的场合（例如切片或是内置的 [`bin()`](https://docs.python.org/zh-cn/3/library/functions.html#bin), [`hex()`](https://docs.python.org/zh-cn/3/library/functions.html#hex) 和 [`oct()`](https://docs.python.org/zh-cn/3/library/functions.html#oct) 函数)。 存在此方法表明数字对象属于整数类型。 必须返回一个整数。
	
	如果未定义 [`__int__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__int__), [`__float__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__float__) 和 [`__complex__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__complex__) 则相应的内置函数 [`int()`](https://docs.python.org/zh-cn/3/library/functions.html#int), [`float()`](https://docs.python.org/zh-cn/3/library/functions.html#float) 和 [`complex()`](https://docs.python.org/zh-cn/3/library/functions.html#complex) 将回退为 [`__index__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__index__)。

- `object.__round__`(*self*[, *ndigits*])

- `object.__trunc__`(*self*)

- `object.__floor__`(*self*)

- `object.__ceil__`(*self*)

    调用这些方法以实现内置函数 [`round()`](https://docs.python.org/zh-cn/3/library/functions.html#round) 以及 [`math`](https://docs.python.org/zh-cn/3/library/math.html#module-math) 函数 [`trunc()`](https://docs.python.org/zh-cn/3/library/math.html#math.trunc), [`floor()`](https://docs.python.org/zh-cn/3/library/math.html#math.floor) 和 [`ceil()`](https://docs.python.org/zh-cn/3/library/math.html#math.ceil)。 除了将 *ndigits* 传给 `__round__()` 的情况之外这些方法的返回值都应当是原对象截断为 [`Integral`](https://docs.python.org/zh-cn/3/library/numbers.html#numbers.Integral) (通常为 [`int`](https://docs.python.org/zh-cn/3/library/functions.html#int))。
	
	如果未定义 [`__int__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__int__) 则内置函数 [`int()`](https://docs.python.org/zh-cn/3/library/functions.html#int) 会回退到 [`__trunc__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__trunc__)。



### 3.3.9. with 语句上下文管理器

*上下文管理器* 是一个对象，它定义了在执行 [`with`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#with) 语句时要建立的运行时上下文。 上下文管理器处理进入和退出所需运行时上下文以执行代码块。 通常使用 `with` 语句（在 [with 语句](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#with) 中描述），但是也可以通过直接调用它们的方法来使用。

上下文管理器的典型用法包括保存和恢复各种全局状态，锁定和解锁资源，关闭打开的文件等等。

要了解上下文管理器的更多信息，请参阅 [上下文管理器类型](https://docs.python.org/zh-cn/3/library/stdtypes.html#typecontextmanager)。

- `object.__enter__`(*self*)

    进入与此对象相关的运行时上下文。 [`with`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#with) 语句将会绑定这个方法的返回值到 `as` 子句中指定的目标，如果有的话。

- `object.__exit__`(*self*, *exc_type*, *exc_value*, *traceback*)

    退出关联到此对象的运行时上下文。 各个参数描述了导致上下文退出的异常。 如果上下文是无异常地退出的，三个参数都将为 [`None`](https://docs.python.org/zh-cn/3/library/constants.html#None)。如果提供了异常，并且希望方法屏蔽此异常（即避免其被传播），则应当返回真值。 否则的话，异常将在退出此方法时按正常流程处理。请注意 [`__exit__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__exit__) 方法不应该重新引发被传入的异常，这是调用者的责任。

参见
- [**PEP 343**](https://www.python.org/dev/peps/pep-0343) - "with" 语句

    Python [`with`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#with) 语句的规范描述、背景和示例。



### 3.3.10. 特殊方法查找

对于自定义类来说，特殊方法的隐式发起调用仅保证在其定义于对象类型中时能正确地发挥作用，而不能定义在对象实例字典中。 该行为就是以下代码会引发异常的原因。:


```shell
>>> class C:
...     pass
...
>>> c = C()
>>> c.__len__ = lambda: 5
>>> len(c)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: object of type 'C' has no len()
```

此行为背后的原理在于包括类型对象在内的所有对象都会实现的几个特殊方法，例如 [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__) 和 [`__repr__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__repr__)。 如果这些方法的隐式查找使用了传统的查找过程，它们会在对类型对象本身发起调用时失败:


```shell
>>> 1 .__hash__() == hash(1)
True
>>> int.__hash__() == hash(int)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: descriptor '__hash__' of 'int' object needs an argument
```

以这种方式不正确地尝试发起调用一个类的未绑定方法有时被称为‘元类混淆’，可以通过在查找特殊方法时绕过实例的方式来避免:


```shell
>>> type(1).__hash__(1) == hash(1)
True
>>> type(int).__hash__(int) == hash(int)
True
```

除了为了正确性而绕过任何实例属性之外，隐式特殊方法查找通常也会绕过 [`__getattribute__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getattribute__) 方法，甚至包括对象的元类:


```shell
>>> class Meta(type):
...     def __getattribute__(*args):
...         print("Metaclass getattribute invoked")
...         return type.__getattribute__(*args)
...
>>> class C(object, metaclass=Meta):
...     def __len__(self):
...         return 10
...     def __getattribute__(*args):
...         print("Class getattribute invoked")
...         return object.__getattribute__(*args)
...
>>> c = C()
>>> c.__len__()                 # Explicit lookup via instance
Class getattribute invoked
10
>>> type(c).__len__(c)          # Explicit lookup via type
Metaclass getattribute invoked
10
>>> len(c)                      # Implicit lookup
10
```

以这种方式绕过 [`__getattribute__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__getattribute__) 机制为解析器内部的速度优化提供了显著的空间，其代价则是牺牲了处理特殊方法时的一些灵活性（特殊方法 *必须* 设置在类对象本身上以便始终一致地由解释器发起调用）。



## 3.4. 协程

### 3.4.1. 可等待对象

[awaitable](https://docs.python.org/zh-cn/3/glossary.html#term-awaitable) 对象主要实现了 [`__await__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__await__) 方法。 从 [`async def`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#async-def) 函数返回的 [协程对象](https://docs.python.org/zh-cn/3/glossary.html#term-coroutine) 即属于可等待对象。

>**注解:** 从带有 [`types.coroutine()`](https://docs.python.org/zh-cn/3/library/types.html#types.coroutine) 或 [`asyncio.coroutine()`](https://docs.python.org/zh-cn/3/library/asyncio-task.html#asyncio.coroutine) 装饰器的生成器返回的 [generator iterator](https://docs.python.org/zh-cn/3/glossary.html#term-generator-iterator) 对象也属于可等待对象，但它们并未实现 [`__await__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__await__)。

- `object.__await__`(*self*)

    必须返回一个 [iterator](https://docs.python.org/zh-cn/3/glossary.html#term-iterator)。 应当被用来实现 [awaitable](https://docs.python.org/zh-cn/3/glossary.html#term-awaitable) 对象。 例如，[`asyncio.Future`](https://docs.python.org/zh-cn/3/library/asyncio-future.html#asyncio.Future) 实现了此方法以与 [`await`](https://docs.python.org/zh-cn/3/reference/expressions.html#await) 表达式相兼容。

*3.5 新版功能.*

参见
[**PEP 492**](https://www.python.org/dev/peps/pep-0492) 了解有关可等待对象的详细信息。



### 3.4.2. 协程对象

[协程对象](https://docs.python.org/zh-cn/3/glossary.html#term-coroutine) 属于 [awaitable](https://docs.python.org/zh-cn/3/glossary.html#term-awaitable) 对象。 协程的执行可通过调用 [`__await__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__await__) 并迭代其结果来控制。 当协程结束执行并返回时，迭代器会引发 [`StopIteration`](https://docs.python.org/zh-cn/3/library/exceptions.html#StopIteration)，该异常的 `value` 属性将存放返回值。 如果协程引发了异常，它会被迭代器所传播。 协程不应直接引发未处理的 [`StopIteration`](https://docs.python.org/zh-cn/3/library/exceptions.html#StopIteration) 异常。

协程也具有下面列出的方法，它们类似于生成器的对应方法 (参见 [生成器-迭代器的方法](https://docs.python.org/zh-cn/3/reference/expressions.html#generator-methods))。 但是，与生成器不同，协程并不直接支持迭代。

*在 3.5.2 版更改:* 等待一个协程超过一次将引发 [`RuntimeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#RuntimeError)。

- `coroutine.send`(*value*)

    开始或恢复协程的执行。 如果 *value* 为 `None`，则这相当于前往 [`__await__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__await__) 所返回迭代器的下一项。 如果 *value* 不为 `None`，此方法将委托给导致协程挂起的迭代器的 [`send()`](https://docs.python.org/zh-cn/3/reference/expressions.html#generator.send) 方法。 其结果（返回值，[`StopIteration`](https://docs.python.org/zh-cn/3/library/exceptions.html#StopIteration) 或是其他异常）将与上述对 [`__await__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__await__) 返回值进行迭代的结果相同。

- `coroutine.throw`(*type*[, *value*[, *traceback*]])

    在协程内引发指定的异常。 此方法将委托给导致协程挂起的迭代器的 [`throw()`](https://docs.python.org/zh-cn/3/reference/expressions.html#generator.throw) 方法，如果存在该方法。 否则的话，异常会在挂起点被引发。 其结果（返回值，[`StopIteration`](https://docs.python.org/zh-cn/3/library/exceptions.html#StopIteration) 或是其他异常）将与上述对 [`__await__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__await__) 返回值进行迭代的结果相同。 如果异常未在协程内被捕获，则将回传给调用者。

- `coroutine.close`()

    此方法会使得协程清理自身并退出。 如果协程被挂起，此方法会先委托给导致协程挂起的迭代器的 [`close()`](https://docs.python.org/zh-cn/3/reference/expressions.html#generator.close) 方法，如果存在该方法。 然后它会在挂起点引发 [`GeneratorExit`](https://docs.python.org/zh-cn/3/library/exceptions.html#GeneratorExit)，使得协程立即清理自身。 最后，协程会被标记为已结束执行，即使它根本未被启动。当协程对象将要被销毁时，会使用以上处理过程来自动关闭。



### 3.4.3. 异步迭代器

*异步迭代器* 可以在其 `__anext__` 方法中调用异步代码。

异步迭代器可在 [`async for`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#async-for) 语句中使用。

- `object.__aiter__`(*self*)

    必须返回一个 *异步迭代器* 对象。

- `object.__anext__`(*self*)

    必须返回一个 *可迭代对象* 输出迭代器的下一结果值。 当迭代结束时应该引发 [`StopAsyncIteration`](https://docs.python.org/zh-cn/3/library/exceptions.html#StopAsyncIteration) 错误。

异步可迭代对象的一个示例:

```python
class Reader:
    async def readline(self):
        ...

    def __aiter__(self):
        return self

    async def __anext__(self):
        val = await self.readline()
        if val == b'':
            raise StopAsyncIteration
        return val
```

*3.5 新版功能.*

*在 3.7 版更改:* 在 Python 3.7 之前，`__aiter__` 可以返回一个 *可迭代对象* 并解析为 [异步迭代器](https://docs.python.org/zh-cn/3/glossary.html#term-asynchronous-iterator)。

从 Python 3.7 开始，`__aiter__` 必须 返回一个异步迭代器对象。 返回任何其他对象都将导致 [`TypeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#TypeError) 错误。



### 3.4.4. 异步上下文管理器

*异步上下文管理器* 是 *上下文管理器* 的一种，它能够在其 `__aenter__` 和 `__aexit__` 方法中暂停执行。

异步上下文管理器可在 [`async with`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#async-with) 语句中使用。

- `object.__aenter__`(*self*)

    在语义上类似于 [`__enter__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__enter__)，仅有的区别是它必须返回一个 *可等待对象*。

- `object.__aexit__`(*self*, *exc_type*, *exc_value*, *traceback*)

    在语义上类似于 [`__exit__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__exit__)，仅有的区别是它必须返回一个 *可等待对象*。

异步上下文管理器类的一个示例:

```python
class AsyncContextManager:
    async def __aenter__(self):
        await log('entering context')

    async def __aexit__(self, exc_type, exc, tb):
        await log('exiting context')
```

*3.5 新版功能.*

备注

- [1](https://docs.python.org/zh-cn/3/reference/datamodel.html#id1)

    在某些情况下 *有可能* 基于可控的条件改变一个对象的类型。 但这通常不是个好主意，因为如果处理不当会导致一些非常怪异的行为。

- [2](https://docs.python.org/zh-cn/3/reference/datamodel.html#id2)

    [`__hash__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__hash__), [`__iter__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__iter__), [`__reversed__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__reversed__) 以及 [`__contains__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__contains__) 方法对此有特殊处理；其他方法仍会引发 [`TypeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#TypeError)，但可能依靠 `None` 属于不可调用对象的行为来做到这一点。

- [3](https://docs.python.org/zh-cn/3/reference/datamodel.html#id5)

    这里的“不支持”是指该类无此方法，或方法返回 `NotImplemented`。 如果你想强制回退到右操作数的反射方法，请不要设置方法为 `None` — 那会造成显式地 *阻塞* 此种回退的相反效果。

- [4](https://docs.python.org/zh-cn/3/reference/datamodel.html#id6)

    对于相同类型的操作数，如果非反射方法 (例如 [`__add__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__add__)) 失败则会认为相应运算不被支持，这就是反射方法未被调用的原因。