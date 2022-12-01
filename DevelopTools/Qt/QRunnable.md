

#qt

#待处理

[toc]

[原文](https://www.cnblogs.com/lcgbk/p/13952117.html)

继承QRunnable+QThreadPool实现多线程的方法个人感觉使用的相对较少，在这里只是简单介绍下使用的方法。我们可以根据使用的场景来选择方法。

**此方法和QThread的区别：** 

*   与外界通信方式不同。由于QThread是继承于QObject的，但QRunnable不是，所以在QThread线程中，可以直接将线程中执行的结果通过信号的方式发到主程序，而QRunnable线程不能用信号槽，只能通过别的方式，等下会介绍；
*   启动线程方式不同。QThread线程可以直接调用start()函数启动，而QRunnable线程需要借助QThreadPool进行启动；
*   资源管理不同。QThread线程对象需要手动去管理删除和释放，而QRunnable则会在QThreadPool调用完成后自动释放。

接下来就来看看QRunnable的用法、使用场景以及注意事项；

要使用QRunnable创建线程，步骤如下：

*   继承QRunnable。和QThread使用一样， 首先需要将你的线程类继承于QRunnable；
*   重写run函数。还是和QThread一样，需要重写run函数；
*   使用QThreadPool启动线程。

继承于QRunnable的类：

```cpp
#ifndef INHERITQRUNNABLE_H
#define INHERITQRUNNABLE_H

#include <QRunnable>
#include <QWidget>
#include <QDebug>
#include <QThread>

class CusRunnable : public QRunnable
{
public:
    explicit CusRunnable()
    {
    }

    ~CusRunnable()
    {
        qDebug() << __FUNCTION__;
    }

    void run()
    {
        qDebug() << __FUNCTION__ << QThread::currentThreadId();
        QThread::msleep(1000);
    }
};

#endif // INHERITQRUNNABLE_H
```

主界面类：

```cpp
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "ui_mainwindow.h"
#include "InheritQRunnable.h"
#include <QThreadPool>
#include <QDebug>

namespace Ui
{
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0) : QMainWindow(parent),
                                               ui(new Ui::MainWindow)
    {
        ui->setupUi(this);

        m_pRunnable = new CusRunnable();
        qDebug() << __FUNCTION__ << QThread::currentThreadId();
        QThreadPool::globalInstance()->start(m_pRunnable);
    }

    ~MainWindow()
    {
        qDebug() << __FUNCTION__;
        delete ui;
    }

private:
    Ui::MainWindow *ui;
    CusRunnable *m_pRunnable = nullptr;
};

#endif // MAINWINDOW_H

```

直接运行以上实例，结果输出如下：

```shell
MainWindow 0x377c
run 0x66ac
~CusRunnable
```

我们可以看到这里打印的线程ID是不同的，说明是在不同线程中执行，而线程执行完后就自动进入到析构函数中， 不需要手动释放。

上面我们说到要启动QRunnable线程，需要QThreadPool配合使用，而调用方式有两种：全局线程池和非全局线程池。

**（1）使用全局线程池启动**

```cpp
QThreadPool::globalInstance()->start(m_pRunnable);
```

**（2）使用非全局线程池启动**

该方式可以控制线程最大数量， 以及其他设置，比较灵活，具体参照帮助文档。

```cpp
QThreadPool threadpool; 
threadpool.setMaxThreadCount(1); 
threadpool.start(m_pRunnable);
```

前面我们提到，因为QRunnable没有继承于QObject，所以没法使用信号槽与外界通信，那么，如果要在QRunnable线程中和外界通信怎么办呢，通常有两种做法：

*   使用多继承。让我们的自定义线程类同时继承于QRunnable和QObject，这样就可以使用信号和槽，但是多线程使用比较麻烦，特别是继承于自定义的类时，容易出现接口混乱，所以在项目中尽量少用多继承。
*   使用QMetaObject::invokeMethod。

接下来只介绍使用QMetaObject::invokeMethod来通信：

**QMetaObject::invokeMethod** 函数定义如下：

```cpp
static bool QMetaObject::invokeMethod(
                         QObject *obj, const char *member,
                         Qt::ConnectionType,
                         QGenericReturnArgument ret,
                         QGenericArgument val0 = QGenericArgument(Q_NULLPTR),
                         QGenericArgument val1 = QGenericArgument(),
                         QGenericArgument val2 = QGenericArgument(),
                         QGenericArgument val3 = QGenericArgument(),
                         QGenericArgument val4 = QGenericArgument(),
                         QGenericArgument val5 = QGenericArgument(),
                         QGenericArgument val6 = QGenericArgument(),
                         QGenericArgument val7 = QGenericArgument(),
                         QGenericArgument val8 = QGenericArgument(),
                         QGenericArgument val9 = QGenericArgument())；

```

该函数就是尝试调用obj的member函数，可以是信号、槽或者Q\_INVOKABLE声明的函数（能够被Qt元对象系统唤起），只需要将函数的名称传递给此函数，调用成功返回true，失败返回false。member函数调用的返回值放在ret中，如果调用是异步的，则不能计算返回值。你可以将最多10个参数（val0、val1、val2、val3、val4、val5、val6、val7、val8和val9）传递给member函数，必须使用Q\_ARG()和Q\_RETURN\_ARG()宏封装参数，Q\_ARG()接受类型名 + 该类型的常量引用；Q\_RETURN\_ARG()接受一个类型名 + 一个非常量引用。

QMetaObject::invokeMethod可以是异步调用，也可以是同步调用。这取决与它的连接方式Qt::ConnectionType type：

*   如果类型是Qt::DirectConnection，则会立即调用该成员，同步调用。
*   如果类型是Qt::QueuedConnection，当应用程序进入主事件循环时，将发送一个QEvent并调用该成员，异步调用。
*   如果类型是Qt::BlockingQueuedConnection，该方法将以与Qt::QueuedConnection相同的方式调用，不同的地方：当前线程将阻塞，直到事件被传递。使用此连接类型在同一线程中的对象之间通信将导致死锁。
*   如果类型是Qt::AutoConnection，如果obj与调用者在同一线程，成员被同步调用；否则，它将异步调用该成员。

我们在主界面中定一个函数，用于更新界面内容：

```cpp
Q_INVOKABLE void setText(QString msg){
    ui->label->setText(msg);
}
```

继承于QRunnable的线程类，修改完成如下：

```cpp
#ifndef INHERITQRUNNABLE_H
#define INHERITQRUNNABLE_H

#include <QRunnable>
#include <QWidget>
#include <QDebug>
#include <QThread>

class CusRunnable : public QRunnable
{
public:
    //修改构造函数
    explicit CusRunnable(QObject *obj):m_pObj(obj){
    }

    ~CusRunnable(){
        qDebug() << __FUNCTION__;
    }

    void run(){
        qDebug() << __FUNCTION__ << QThread::currentThreadId();
        QMetaObject::invokeMethod(m_pObj,"setText",Q_ARG(QString,"hello world!")); //此处与外部通信
        QThread::msleep(1000);
    }

private:
    QObject * m_pObj = nullptr; //定义指针
};

#endif // INHERITQRUNNABLE_H
```

创建线程对象时，需要将主界面对象传入线程类，如下：

```cpp
m_pRunnable = new CusRunnable(this);
```

到这里也就实现了线程与外部通信了，运行效果如下：

[![](https://img2020.cnblogs.com/blog/2085020/202011/2085020-20201110090834920-1274689039.png)
](https://img2020.cnblogs.com/blog/2085020/202011/2085020-20201110090834920-1274689039.png)

*   使用该方法实现的多线程，线程中的资源无需用户手动释放，线程执行完后会自动回收资源；
*   和继承QThread的方法一样需要继承类，并且重新实现run函数；
*   需要结合QThreadPool线程池来使用；
*   与外界通信可以使用如果使用信号槽机制会比较麻烦，可以使用QMetaObject::invokeMethod的方式与外界通信。

_**本文章实例的源码地址：[https://gitee.com/CogenCG/QThreadExample.git](https://gitee.com/CogenCG/QThreadExample.git)**_