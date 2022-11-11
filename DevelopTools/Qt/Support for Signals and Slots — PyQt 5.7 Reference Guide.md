#python 
#qt 

[原文](https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html)

[toc]
Qt 的一个关键特性是它使用信号和插槽在对象之间进行通信。它们的使用鼓励了可重用组件的开发。

A signal is emitted when something of potential interest happens. A slot is a Python callable. If a signal is connected to a slot then the slot is called when the signal is emitted. If a signal isn’t connected then nothing happens. The code (or component) that emits the signal does not know or care if the signal is being used.

当一些潜在的感兴趣的事情发生时，信号就会发出。插槽是 Python 可调用的。如果一个信号连接到一个插槽，然后该插槽被调用时，信号被发射。如果一个信号没有连接，那么什么也不会发生。发出信号的代码(或组件)不知道或不关心信号是否被使用。

The signal/slot mechanism has the following features.

信号/槽机构具有以下特点。

*   A signal may be connected to many slots. 一个信号可以连接到多个插槽
*   A signal may also be connected to another signal. 一个信号也可以连接到另一个信号
*   Signal arguments may be any Python type. 信号参数可以是任何 Python 类型
*   A slot may be connected to many signals. 一个插槽可以连接到许多信号
*   Connections may be direct (ie. synchronous) or queued (ie. asynchronous). 连接可以是直接(例如同步)或排队(例如异步)
*   Connections may be made across threads. 可以跨线程进行连接
*   Signals may be disconnected. 信号可能被切断

Unbound and Bound Signals 无约束信号和有约束信号[¶](#unbound-and-bound-signals "Permalink to this headline")
-------------------------------------------------------------------------------------------------

A signal (specifically an unbound signal) is a class attribute. When a signal is referenced as an attribute of an instance of the class then PyQt5 automatically binds the instance to the signal in order to create a _bound signal_. This is the same mechanism that Python itself uses to create bound methods from class functions.

信号(特别是未绑定信号)是一个类属性。当信号被引用为类的实例的属性时，pyqt5会自动将实例绑定到信号，以创建绑定信号。这与 Python 本身使用的从类函数创建绑定方法的机制相同。

A bound signal has `connect()`, `disconnect()` and `emit()` methods that implement the associated functionality. It also has a `signal` attribute that is the signature of the signal that would be returned by Qt’s `SIGNAL()` macro.

绑定信号具有实现相关功能的 connect ()、 disconnect ()和 emit ()方法。它还有一个信号属性，即 Qt 的 SIGNAL ()宏将返回的信号的签名。

A signal may be overloaded, ie. a signal with a particular name may support more than one signature. A signal may be indexed with a signature in order to select the one required. A signature is a sequence of types. A type is either a Python type object or a string that is the name of a C++ type. The name of a C++ type is automatically normalised so that, for example, `QVariant` can be used instead of the non-normalised `const QVariant &`.

信号可能过载。具有特定名称的信号可以支持多个签名。可以用签名对信号进行索引，以便选择所需的签名。签名是一系列类型。类型可以是 Python 类型对象，也可以是 c + + 类型名称的字符串。C + + 类型的名称自动标准化，以便，例如，QVariant 可以用来代替非标准化的 const QVariant & 。

If a signal is overloaded then it will have a default that will be used if no index is given.

如果一个信号是重载的，那么它将有一个默认值，如果没有给出索引将被使用。

When a signal is emitted then any arguments are converted to C++ types if possible. If an argument doesn’t have a corresponding C++ type then it is wrapped in a special C++ type that allows it to be passed around Qt’s meta-type system while ensuring that its reference count is properly maintained.

当一个信号被发出时，如果可能的话，任何参数都被转换成 c + + 类型。如果一个参数没有对应的 c + + 类型，那么它就被包装在一个特殊的 c + + 类型中，这个类型允许它在 Qt 的元类型系统中传递，同时确保它的引用计数得到正确的维护。

Defining New Signals with 定义新信号[`pyqtSignal()`](#PyQt5.QtCore.pyqtSignal "PyQt5.QtCore.pyqtSignal")[¶](#defining-new-signals-with-pyqtsignal "Permalink to this headline")
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

PyQt5 automatically defines signals for all Qt’s built-in signals. New signals can be defined as class attributes using the [`pyqtSignal()`](#PyQt5.QtCore.pyqtSignal "PyQt5.QtCore.pyqtSignal") factory.

Pyqt5自动为所有 Qt 内置信号定义信号。可以使用 pyqtSignal ()工厂将新信号定义为类属性。

`PyQt5.QtCore.``pyqtSignal`(_types 类型_\[, _name 名称_\[, _revision=0 0_\[, _arguments=\[\] 参数 = \[\]_\]\]\])[¶](#PyQt5.QtCore.pyqtSignal "Permalink to this definition")

Create one or more overloaded unbound signals as a class attribute.

创建一个或多个重载的未绑定信号作为类属性。

  
| Parameters: 参数: | 

*   **types 类型** – the types that define the C++ signature of the signal. Each type may be a Python type object or a string that is the name of a C++ type. Alternatively each may be a sequence of type arguments. In this case each sequence defines the signature of a different signal overload. The first overload will be the default. - 定义信号的 c + + 签名的类型。每个类型可以是 Python 类型的对象，也可以是 c + + 类型的名称字符串。或者，每个参数都可以是一个类型参数序列。在这种情况下，每个序列定义了不同信号过载的签名。第一个重载将是默认的
*   **name 名称** – the name of the signal. If it is omitted then the name of the class attribute is used. This may only be given as a keyword argument. - 信号的名称。如果省略，则使用 class 属性的名称。这只能作为关键字参数给出
*   **revision 修订** – the revision of the signal that is exported to QML. This may only be given as a keyword argument. - 输出至 QML 的信号的修订。这只能作为关键字参数给出
*   **arguments 争论** – the sequence of the names of the signal’s arguments that is exported to QML. This may only be given as a keyword argument. - 输出到 QML 的信号参数的名称序列。这只能作为关键字参数给出

 |
| Return type: 返回类型: | 

an unbound signal

无约束信号

 |

The following example shows the definition of a number of new signals:

下面的例子显示了一些新信号的定义:

from PyQt5.QtCore import QObject, pyqtSignal

class Foo(QObject):

    \# This defines a signal called 'closed' that takes no arguments.
    closed \= pyqtSignal()

    \# This defines a signal called 'rangeChanged' that takes two
    \# integer arguments.
    range\_changed \= pyqtSignal(int, int, name\='rangeChanged')

    \# This defines a signal called 'valueChanged' that has two overloads,
    \# one that takes an integer argument and one that takes a QString
    \# argument.  Note that because we use a string to specify the type of
    \# the QString argument then this code will run under Python v2 and v3.
    valueChanged \= pyqtSignal(\[int\], \['QString'\])

New signals should only be defined in sub-classes of [`QObject`](https://doc.bccnsoft.com/docs/PyQt5/api/qobject.html#PyQt5.QtCore.QObject "PyQt5.QtCore.QObject"). They must be part of the class definition and cannot be dynamically added as class attributes after the class has been defined.

新的信号应该只在 QObject 的子类中定义。它们必须是类定义的一部分，并且在定义了类之后不能作为类属性动态添加。

New signals defined in this way will be automatically added to the class’s [`QMetaObject`](https://doc.bccnsoft.com/docs/PyQt5/api/qmetaobject.html#PyQt5.QtCore.QMetaObject "PyQt5.QtCore.QMetaObject"). This means that they will appear in Qt Designer and can be introspected using the [`QMetaObject`](https://doc.bccnsoft.com/docs/PyQt5/api/qmetaobject.html#PyQt5.QtCore.QMetaObject "PyQt5.QtCore.QMetaObject") API.

以这种方式定义的新信号将自动添加到类的 QMetaObject 中。这意味着它们将出现在 Qt 设计器中，并且可以使用 QMetaObject API 进行自省。

Overloaded signals should be used with care when an argument has a Python type that has no corresponding C++ type. PyQt5 uses the same internal C++ class to represent such objects and so it is possible to have overloaded signals with different Python signatures that are implemented with identical C++ signatures with unexpected results. The following is an example of this:

当参数的 Python 类型没有对应的 c + + 类型时，应当谨慎使用重载信号。Pyqt5使用相同的内部 c + + 类来表示这些对象，因此可以使用不同的 Python 签名重载信号，这些信号使用相同的 c + + 签名来实现，并得到意外的结果。下面是一个例子:

class Foo(QObject):

    \# This will cause problems because each has the same C++ signature.
    valueChanged \= pyqtSignal(\[dict\], \[list\])

Connecting, Disconnecting and Emitting Signals 连接、断开和发射信号[¶](#connecting-disconnecting-and-emitting-signals "Permalink to this headline")
-----------------------------------------------------------------------------------------------------------------------------------------

Signals are connected to slots using the [`connect()`](#connect "connect") method of a bound signal.

使用绑定信号的 connect ()方法将信号连接到槽。

`connect`(_slot 槽_\[, _type=PyQt5.QtCore.Qt.AutoConnection_\[, _no\_receiver\_check=False 无接收器检查 = False_\]\])[¶](#connect "Permalink to this definition")

Connect a signal to a slot. An exception will be raised if the connection failed.

将信号连接到插槽。如果连接失败将引发异常。

  
| Parameters: 参数: | 

*   **slot 槽** – the slot to connect to, either a Python callable or another bound signal. - 要连接到的插槽，可以是 Python 可调用的，也可以是其他绑定信号
*   **type 类型** – the type of the connection to make. - 连接的类型
*   **no\_receiver\_check 没有接收器，检查** – suppress the check that the underlying C++ receiver instance still exists and deliver the signal anyway. - 取消底层 c + + 接收器实例是否仍然存在的检查，并无论如何发送信号

 |

Signals are disconnected from slots using the [`disconnect()`](#disconnect "disconnect") method of a bound signal.

使用绑定信号的断开()方法将信号从槽中断开。

`disconnect`(\[_slot 槽_\])[¶](#disconnect "Permalink to this definition")

Disconnect one or more slots from a signal. An exception will be raised if the slot is not connected to the signal or if the signal has no connections at all.

从信号断开一个或多个插槽。如果槽没有连接到信号或者信号根本没有连接，则会引发异常。

  
| Parameters: 参数: | **slot 槽** – the optional slot to disconnect from, either a Python callable or another bound signal. If it is omitted then all slots connected to the signal are disconnected. - 断开连接的可选插槽，可以是 Python 可调用的，也可以是其他绑定信号。如果它被省略，那么所有连接到信号的插槽都断开 |

Signals are emitted from using the [`emit()`](#emit "emit") method of a bound signal.

使用绑定信号的发射()方法发射信号。

`emit`(_\*args \* args_)[¶](#emit "Permalink to this definition")

Emit a signal.

发出信号。

  
| Parameters: 参数: | **args 方舟** – the optional sequence of arguments to pass to any connected slots. - 传递到任何连接插槽的可选参数序列 |

The following code demonstrates the definition, connection and emit of a signal without arguments:

下面的代码演示了没有参数的信号的定义、连接和发射:

from PyQt5.QtCore import QObject, pyqtSignal

class Foo(QObject):

    \# Define a new signal called 'trigger' that has no arguments.
    trigger \= pyqtSignal()

    def connect\_and\_emit\_trigger(self):
        \# Connect the trigger signal to a slot.
        self.trigger.connect(self.handle\_trigger)

        \# Emit the signal.
        self.trigger.emit()

    def handle\_trigger(self):
        \# Show that the slot has been called.

        print "trigger signal received"

The following code demonstrates the connection of overloaded signals:

下面的代码演示了重载信号的连接:

from PyQt5.QtWidgets import QComboBox

class Bar(QComboBox):

    def connect\_activated(self):
        \# The PyQt5 documentation will define what the default overload is.
        \# In this case it is the overload with the single integer argument.
        self.activated.connect(self.handle\_int)

        \# For non-default overloads we have to specify which we want to
        \# connect.  In this case the one with the single string argument.
        \# (Note that we could also explicitly specify the default if we
        \# wanted to.)
        self.activated\[str\].connect(self.handle\_string)

    def handle\_int(self, index):
        print "activated signal passed integer", index

    def handle\_string(self, text):
        print "activated signal passed QString", text

Connecting Signals Using Keyword Arguments 使用关键字参数连接信号[¶](#connecting-signals-using-keyword-arguments "Permalink to this headline")
-----------------------------------------------------------------------------------------------------------------------------------

It is also possible to connect signals by passing a slot as a keyword argument corresponding to the name of the signal when creating an object, or using the `pyqtConfigure()` method. For example the following three fragments are equivalent:

在创建对象时，还可以通过传递 slot 作为与信号名称对应的关键字参数来连接信号，或者使用 pyqtConfigure ()方法。例如，下面三个片段是等价的:

act \= QAction("Action", self)
act.triggered.connect(self.on\_triggered)

act \= QAction("Action", self, triggered\=self.on\_triggered)

act \= QAction("Action", self)
act.pyqtConfigure(triggered\=self.on\_triggered)

The 这个[`pyqtSlot()`](#PyQt5.QtCore.pyqtSlot "PyQt5.QtCore.pyqtSlot") Decorator 室内设计师[¶](#the-pyqtslot-decorator "Permalink to this headline")
---------------------------------------------------------------------------------------------------------------------------------------------

Although PyQt5 allows any Python callable to be used as a slot when connecting signals, it is sometimes necessary to explicitly mark a Python method as being a Qt slot and to provide a C++ signature for it. PyQt5 provides the [`pyqtSlot()`](#PyQt5.QtCore.pyqtSlot "PyQt5.QtCore.pyqtSlot") function decorator to do this.

尽管 pyqt5允许任何 Python 调用在连接信号时用作插槽，但有时需要显式地将 Python 方法标记为 Qt 插槽，并为其提供 c + + 签名。Pyqt5提供了 pyqtSlot ()函数 decorator 来实现这一点。

`PyQt5.QtCore.``pyqtSlot`(_types 类型_\[, _name 名称_\[, _result 结果_\[, _revision=0 0_\]\]\])[¶](#PyQt5.QtCore.pyqtSlot "Permalink to this definition")

Decorate a Python method to create a Qt slot.

修饰 Python 方法以创建 Qt 插槽。

  
| Parameters: 参数: | 

*   **types 类型** – the types that define the C++ signature of the slot. Each type may be a Python type object or a string that is the name of a C++ type. - 定义插槽的 c + + 签名的类型。每个类型可以是 Python 类型的对象，也可以是 c + + 类型的名称字符串
*   **name 名称** – the name of the slot that will be seen by C++. If omitted the name of the Python method being decorated will be used. This may only be given as a keyword argument. - c + + 将看到的插槽的名称。如果省略，将使用正在修饰的 Python 方法的名称。这只能作为关键字参数给出
*   **revision 修订** – the revision of the slot that is exported to QML. This may only be given as a keyword argument. - 输出至 QML 的插槽的修订。这只能作为关键字参数给出
*   **result 结果** – the type of the result and may be a Python type object or a string that specifies a C++ type. This may only be given as a keyword argument. - 结果的类型，可能是 Python 类型的对象或指定 c + + 类型的字符串。这只能作为关键字参数给出

 |

Connecting a signal to a decorated Python method also has the advantage of reducing the amount of memory used and is slightly faster.

将信号连接到修饰的 Python 方法还具有减少内存使用量的优点，并且速度稍快。

For example:

例如:

from PyQt5.QtCore import QObject, pyqtSlot

class Foo(QObject):

    @pyqtSlot()
    def foo(self):
        """ C++: void foo() """

    @pyqtSlot(int, str)
    def foo(self, arg1, arg2):
        """ C++: void foo(int, QString) """

    @pyqtSlot(int, name\='bar')
    def foo(self, arg1):
        """ C++: void bar(int) """

    @pyqtSlot(int, result\=int)
    def foo(self, arg1):
        """ C++: int foo(int) """

    @pyqtSlot(int, QObject)
    def foo(self, arg1):
        """ C++: int foo(int, QObject \*) """

It is also possible to chain the decorators in order to define a Python method several times with different signatures. For example:

还可以将 decorator 链接起来，以便使用不同的签名多次定义 Python 方法。例如:

from PyQt5.QtCore import QObject, pyqtSlot

class Foo(QObject):

    @pyqtSlot(int)
    @pyqtSlot('QString')
    def valueChanged(self, value):
        """ Two slots will be defined in the QMetaObject. """

The 这个`PyQt_PyObject` Signal Argument Type 信号参数类型[¶](#the-pyqt-pyobject-signal-argument-type "Permalink to this headline")
--------------------------------------------------------------------------------------------------------------------------

It is possible to pass any Python object as a signal argument by specifying `PyQt_PyObject` as the type of the argument in the signature. For example:

通过指定 PyQt \_ pyobject 作为签名中参数的类型，可以将任何 Python 对象作为信号参数传递。例如:

finished \= pyqtSignal('PyQt\_PyObject')

This would normally be used for passing objects where the actual Python type isn’t known. It can also be used to pass an integer, for example, so that the normal conversions from a Python object to a C++ integer and back again are not required.

这通常用于在不知道实际 Python 类型的情况下传递对象。例如，它还可以用于传递整数，因此不需要从 Python 对象到 c + + 整数再返回的常规转换。

The reference count of the object being passed is maintained automatically. There is no need for the emitter of a signal to keep a reference to the object after the call to `finished.emit()`, even if a connection is queued.

被传递对象的引用计数会自动维护。在调用 finished.emit ()之后，信号发射器不需要保留对对象的引用，即使连接已经排队。

Connecting Slots By Name 按名称连接插槽[¶](#connecting-slots-by-name "Permalink to this headline")
-------------------------------------------------------------------------------------------

PyQt5 supports the `connectSlotsByName()` function that is most commonly used by **pyuic5** generated Python code to automatically connect signals to slots that conform to a simple naming convention. However, where a class has overloaded Qt signals (ie. with the same name but with different arguments) PyQt5 needs additional information in order to automatically connect the correct signal.

Pyqt5支持 connectSlotsByName ()函数，该函数最常被 pyuic5生成的 Python 代码使用，用于将信号自动连接到符合简单变数命名原则的插槽。但是，当一个类重载了 Qt 信号时(例如。Pyqt5需要额外的信息，以便自动连接正确的信号。

For example the [`QSpinBox`](https://doc.bccnsoft.com/docs/PyQt5/api/qspinbox.html#PyQt5.QtWidgets.QSpinBox "PyQt5.QtWidgets.QSpinBox") class has the following signals:

例如，QSpinBox 类有以下信号:

void valueChanged(int i);
void valueChanged(const QString &text);

When the value of the spin box changes both of these signals will be emitted. If you have implemented a slot called `on_spinbox_valueChanged` (which assumes that you have given the `QSpinBox` instance the name `spinbox`) then it will be connected to both variations of the signal. Therefore, when the user changes the value, your slot will be called twice - once with an integer argument, and once with a string argument.

当自旋盒的值发生变化时，这两种信号都会被发出。如果您已经实现了一个名为 \_ spinbox \_ valuechanged 的 slot (假设您已经为 QSpinBox 实例提供了名称 spinbox) ，那么它将连接到信号的两种变体。因此，当用户更改值时，将使用整数参数调用 slot 两次，使用字符串参数调用一次。

The [`pyqtSlot()`](#PyQt5.QtCore.pyqtSlot "PyQt5.QtCore.pyqtSlot") decorator can be used to specify which of the signals should be connected to the slot.

可以使用 pyqtSlot () decorator 来指定哪些信号应该连接到插槽。

For example, if you were only interested in the integer variant of the signal then your slot definition would look like the following:

例如，如果您只对信号的整数变量感兴趣，那么您的槽定义如下所示:

@pyqtSlot(int)
def on\_spinbox\_valueChanged(self, i):
    \# i will be an integer.
    pass

If you wanted to handle both variants of the signal, but with different Python methods, then your slot definitions might look like the following:

如果你想同时处理两种不同的信号，但是使用不同的 Python 方法，那么你的插槽定义可能看起来如下所示:

@pyqtSlot(int, name\='on\_spinbox\_valueChanged')
def spinbox\_int\_value(self, i):
    \# i will be an integer.
    pass

@pyqtSlot(str, name\='on\_spinbox\_valueChanged')
def spinbox\_qstring\_value(self, s):
    \# s will be a Python string object (or a QString if they are enabled).
    pass
