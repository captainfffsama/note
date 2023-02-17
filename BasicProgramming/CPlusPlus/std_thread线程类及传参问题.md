#CPP 
#并发 
#CPP标准库


[toc]
[原文](https://www.cnblogs.com/5iedu/p/11633683.html)

# 一. std::thread类

## (一)thread类摘要及分析

```cpp
class thread { // class for observing and managing threads
public:
    class id;

    using native_handle_type = void*;

    thread() noexcept : _Thr{} { // 创建空的thread对象，实际上线程并未被创建！
    }

private:
    template <class _Tuple, size_t... _Indices>
    static unsigned int __stdcall _Invoke(void* _RawVals) noexcept { // enforces termination
        //接口适配：将用户的可调用对象与_beginthreadex的接口进行适配。

        //子线程重新拥有从主线程转让过来的保存着thread参数副本的tuple堆对象的所有权。
        const unique_ptr<_Tuple> _FnVals(static_cast<_Tuple*>(_RawVals));
        _Tuple& _Tup = *_FnVals;
        _STD invoke(_STD move(_STD get<_Indices>(_Tup))...); //注意，由于tuple中保存的都是副本，因此所有的参数都以右值的方式被转发出去。
        _Cnd_do_broadcast_at_thread_exit(); // TRANSITION, ABI
        return 0;
    }

    template <class _Tuple, size_t... _Indices>
    _NODISCARD static constexpr auto _Get_invoke(
        index_sequence<_Indices...>) noexcept { // select specialization of _Invoke to use
        return &_Invoke<_Tuple, _Indices...>;   //这里返回特化的_Invoke函数指针
    }

public:
    template <class _Fn, class... _Args, class = enable_if_t<!is_same_v<_Remove_cvref_t<_Fn>, thread>>>
    explicit thread(_Fn&& _Fx, _Args&& ... _Ax) { // construct with _Fx(_Ax...)
        using _Tuple                 = tuple<decay_t<_Fn>, decay_t<_Args>...>; //将传入thread的所有参数保存着tuple

        //在堆上创建tuple以按值保存thread所有参数的副本，指针用unique_ptr来管理。
        auto _Decay_copied = _STD make_unique<_Tuple>(_STD forward<_Fn>(_Fx), _STD forward<_Args>(_Ax)...); //创建tuple的智能指针
        constexpr auto _Invoker_proc = _Get_invoke<_Tuple>(make_index_sequence<1 + sizeof...(_Args)>{}); //获取线程函数地址

        //在Windows系统中，会调用_beginthredex来创建新线程。其中，_Invoker_proc为线程函数地址，它要求的参数为tuple的指针，即_Decay_copied.get()
        //注意：线程创建后即立即运行(第5个参数为0)，原生的线程id保存在_Thr._Id中，句柄保存在_Thr._Hnd。
        _Thr._Hnd =
            reinterpret_cast<void*>(_CSTD _beginthreadex(nullptr, 0, _Invoker_proc, _Decay_copied.get(), 0, &_Thr._Id));
        if (_Thr._Hnd == nullptr) { // failed to start thread
            _Thr._Id = 0;
            _Throw_Cpp_error(_RESOURCE_UNAVAILABLE_TRY_AGAIN);
        }
        else { // ownership transferred to the thread
            (void)_Decay_copied.release(); //转让tuple的所有权给新的线程。
        }
    }

    ~thread() noexcept { // clean up
        if (joinable()) {  //注意，std::thread析构时，如果线程仍可joinable，则会调用terminate终止程序！
            _STD terminate();
        }
    }

    thread(thread&& _Other) noexcept : _Thr(_STD exchange(_Other._Thr, {})) { // move from _Other
    }

    thread& operator=(thread&& _Other) noexcept { // move from _Other

        if (joinable()) {
            _STD terminate();
        }

        _Thr = _STD exchange(_Other._Thr, {});
        return *this;
    }

    thread(const thread&) = delete;    //thread对象不能被复制
    thread& operator=(const thread&) = delete; //thread对象不能被拷贝赋值

    void swap(thread& _Other) noexcept { // swap with _Other
        _STD swap(_Thr, _Other._Thr);
    }

    _NODISCARD bool joinable() const noexcept { // return true if this thread can be joined
        return _Thr._Id != 0; //原生的线程id不为0，表示底层的线程己经创建
    }

    void join() { // join thread
        if (!joinable()) {
            _Throw_Cpp_error(_INVALID_ARGUMENT);
        }

        if (_Thr._Id == _Thrd_id()) {
            _Throw_Cpp_error(_RESOURCE_DEADLOCK_WOULD_OCCUR);
        }

        if (_Thrd_join(_Thr, nullptr) != _Thrd_success) {
            _Throw_Cpp_error(_NO_SUCH_PROCESS);
        }

        _Thr = {}; //注意调用join以后，原生线程id被清零，意味着join只能被调用一次！
    }

    void detach() { // detach thread
        if (!joinable()) {
            _Throw_Cpp_error(_INVALID_ARGUMENT);
        }

        _Check_C_return(_Thrd_detach(_Thr)); //线程被分离，成为后台线程
        _Thr = {};  //注意调用detach以后，原生线程id被清零，意味着detach也只能被调用一次！
    }

    _NODISCARD id get_id() const noexcept;

    _NODISCARD static unsigned int hardware_concurrency() noexcept { // return number of hardware thread contexts
        return _Thrd_hardware_concurrency();
    }

    _NODISCARD native_handle_type native_handle() { // return Win32 HANDLE as void *
        return _Thr._Hnd;
    }

private:
    _Thrd_t _Thr;
};

```

　　1. 构造std::thread对象时：如果**不带参则会创建一个空的thread对象，但底层线程并没有真正被创建，一般可将其他std::thread对象通过move移入其中**；如果**带参则会创建新线程**，而且会被**立即运行**。

　　2. 在创建thread对象时，std::thread构建函数中的**所有参数均会按值**并**以副本的形式保存成一个tuple对象**。**该tuple由调用线程（一般是主线程）在堆上创建，并交由子线程管理，在子线程结束时同时被释放**。

　　3. joinable()：用于判断std::thread对象联结状态，一个std::thread对象只可能处于可联结或不可联结两种状态之一。
　　
		1). **可联结：当线程己运行或可运行、或处于阻塞时是可联结的**。注意，如果某个底层线程**已经执行完任务**，**但是没有被join的话**，**仍然处于joinable状态**。即std::thread对象与底层线程保持着关联时，为joinable状态。
		2). **不可联结：** 
		- 当**不带参构造的std::thread对象**为不可联结，因为底层线程还没创建。
		- 已移动的std::thread对象为不可联结。
		- **已调用join或detach的对象为不可联结状态**。因为调用join()以后，底层线程已结束，而detach()会把std::thread对象和对应的底层线程之间的连接断开。

　　4. std::thread对象析构时，会先判断是否可joinable()，如果可联结，则会程序会直接被终止。**这意味着创建thread对象以后，要在随后的某个地方调用join或detach以便让std::thread处于不可联结状态**。

　　5. std::thread对象不能被复制和赋值，只能被移动。

## (二)线程的基本用法

### 1. 获取当前信息

　　（1）线程ID：t.get_id();  //其中t为std::thread对象。

　　（2）线程句柄：t.native_handle() //返回与操作系统相关的线程句柄。

　　（3）获取CPU核数：std::thread::hardware_concurrency()，失败时返回0。

### 2. 线程等待和分离

　　（1）join()：等待子线程，调用线程处于阻塞模式

　　（2）detach()：分离子线程，与当前线程的连接被断开，子线程成为后台线程，被C++运行时库接管。

　　（3）joinable()：检查线程是否可被联结。

## (三)`std::this_thread`命名空间中相关辅助函数

　　1. get_id(); //获取线程ID：

　　2. yield(); //当前线程放弃执行，操作系统转去调度另一线程。

　　3. `sleep_until(const xtime* _Abs_time)`：线程休眠至某个指定的时刻`(time point)`，该线程才被重新唤醒。

　　4. `sleep_for(std::chrono::seconds(3))`;//睡眠3秒后才被重新唤醒，不过由于线程调度等原因，实际休眠时间可能比`sleep_duration` 所表示的时间片更长。

【编程实验】std::thread的基本用法

```cpp
#include <iostream>
#include <thread>
#include <chrono>  //for std::chrono::seconds
#include <ctime>   //for std::time_t
#include <iomanip> //for std::put_time

using namespace std;
using namespace std::chrono;   

void thread_func(int x)
{
    cout <<"thread_func start..." << endl;
    cout << "x = " << x << endl;
    cout << "child thread id: " << std::this_thread::get_id() << endl;

    std::this_thread::yield(); //当前线程放弃执行

    cout <<"thread_func end." << endl;
}

void test_sleepUntil()
{
    std::cout <<"thread id " << std::this_thread::get_id() << "'s sleepUntil begin..." << endl;
    using std::chrono::system_clock;
    std::time_t tStart = system_clock::to_time_t(system_clock::now()); //to_time_t：将time_point转为std::time_t
    struct std::tm tm;
    localtime_s(&tm,&tStart);

    std::cout << "Current time: " << std::put_time(&tm, "%X") << std::endl; //X须大写，若小写输出日期
    std::cout << "Waiting for the next minute..." << std::endl;
    
    ++tm.tm_min;
    tm.tm_sec = 0;
    std::this_thread::sleep_until(system_clock::from_time_t(mktime(&tm))); //from_time_t：将time_t转为time_point

    std::cout << std::put_time(&tm, "%X") <<" reach."<<  std::endl; 

    std::cout << "thread id " << std::this_thread::get_id() << "'s sleepUntil end." << endl;
}

int main()
{
    //1. 获取当前线程信息
    cout << "hardware_concurrency: " << std::thread::hardware_concurrency() << endl; //8，当前cpu核数
    cout << "main thread id: " <<std::this_thread::get_id() << endl; //当前线程（主线程）id

    std::thread t(thread_func, 5);
    cout <<"child thread id: " <<t.get_id() << endl; //子线程id
    cout << "child thread handle: " << t.native_handle() << endl;

    //2.joinable检查
    cout << endl;
    std::this_thread::sleep_for(std::chrono::seconds(3)); //主线程睡眠3秒，等待子线程结束

    if (t.joinable()) 
        cout << "t is joinable" << endl;   //该行打印，说明子线程己结束时，仍处于joinable状态！！！
    else 
        cout << "t is unjoinable" << endl;

    t.join();

    //sleep_until
    cout << endl;
    std::thread t2(test_sleepUntil);
    t2.join();

    //传入lambda
    cout << endl;
    std::thread t3([]() {cout <<"t3(thread id: " << std::this_thread::get_id()<< ") is running..." << endl; });
    t3.join();

    return 0;
}
/*输出结果
hardware_concurrency: 8
main thread id: 17672
child thread id: 8172
child thread handle: 000000E4

thread_func start...
x = 5
child thread id: 8172
thread_func end.
t is joinable

thread id 8016's sleepUntil begin...
Current time: 23:21:25
Waiting for the next minute...
23:22:00 reach.
thread id 8016's sleepUntil end.

t3(thread id: 2880) is running...
*/
```

# 二. 传递参数的方式

## (一)传参中的陷阱：

　　1. 向std::thread 构造函数传参：所有参数（含第1个参数可调用对象）均**按值**并**以副本的形式保存**在std::thread对象中的tuple里。这一点的实现类似于std::bind。如果**要达到按引用传参的效果，可使用std::ref来传递**。

　　2. 向线程函数的传参：由于std::thread对象里保存的是参数的副本，为了效率同时兼顾一些只移动类型的对象，**所有的副本均被std::move到线程函数，即以右值的形式传入**。

## (二)注意事项

　　1. 一个实参从主线程传递到子线程的线程函数中，**需要经过两次传递**。**第1次发生在std::thread构造时，此次参数按值并以副本形式被保存**。**第2次发生在向线程函数传递时**，此次传递是由子线程发起，并将之前std::thread内部保存的副本**以右值的形式(std::move())传入线程函数**中的。

　　2. **如果线程函数的形参为T、const T&或T&&类型时**，std::thread的构造函数可以接受左值或右值实参。因为不管是左值还是右值，在std::thread中均是以副本形式被保存，并在第2次向线程函数传参时以右值方式传入，而以上三种形参均可接受右值。

　　3. **而如果线程函数的形参为T&时**，**不管是左值还是右值的T类型实参，都是无法直接经std::thread传递给形参为T&的线程函数**，因为该实参数的副本会被std::move成右值并传递线程函数，但T&无法接受右值类型。**因此，需要以std::ref形式传入（具体原理见下面《编程实验》中的注释）**。

　　4. 当向线程函数传参时，可能发生隐式类型转换，这种转换是在子线程中进行的。需要注意，由于隐式转换会构造临时对象，并将该对象（是个右值）传入线程函数，因此线程函数的形参应该是可接受右值类型的T、const T&或T&&类型，但不能是T&类型。此外，如果源类型是指针或引用类型时，还要防止可能发生悬空指针和悬空引用的现象。

【编程实验】std::thread传参中的陷阱

```cpp
#include <iostream>
#include <thread>
#include <chrono>

using namespace std;
using namespace std::chrono;   //for std::chrono::seconds

class Widget 
{
public:
    mutable int mutableInt = 0;

    //Widget() :mutableInt(0) {}
    Widget() : mutableInt(0) { cout << "Widget(), thread id = "<< std::this_thread::get_id() << endl;}

    //类型转换构造函数
    Widget(int i):mutableInt(i){ cout << "Widget(int i), thread id = " << std::this_thread::get_id() << endl; }

    Widget(const Widget& w):mutableInt(w.mutableInt) { cout << "Widget(const Widget& w), thread id = " << std::this_thread::get_id() << endl; }
    Widget(Widget&& w)  noexcept  //移动构造
    { 
        mutableInt = w.mutableInt; 
        cout << "Widget(Widget && w), thread id = " << std::this_thread::get_id() << endl;
    }

    void func(const string& s) { cout <<"void func(string& s),  thread id = " << std::this_thread::get_id() << endl; }
};

void updateWidget_implicit(const Widget& w)
{
    cout << "invoke updateWidget_implicit, thread id =" << std::this_thread::get_id() << endl;
}

void updateWidget_ref(Widget& w)
{
    cout << "invoke updateWidget_ref, thread id =" << std::this_thread::get_id() << endl;
}

void updateWidget_cref(const Widget& w)
{
    cout << "invoke updateWidget_cref, thread id =" << std::this_thread::get_id() << endl;
}

void test_ctor(const Widget& w) //注意这里的w是按引用方式传入（引用的是std::thread中保存的参数副本）
{
    cout << "thread begin...(id = " << std::this_thread::get_id() << ")" << endl;
    cout << "w.matableInt = " << ++w.mutableInt << endl;//注意，当std::thread按值传参时，此处修改的是std::thread中
                                                        //保存的参数副本，而不是main中的w。
                                                        //而当向std::thread按std::ref传参时，先会创建一个std::ref临时对象，
                                                        //其中保存着main中w引用。然后这个std::ref再以副本的形式保存在
                                                        //std::thread中。随后这个副本被move到线程函数，由于std::ref重载了
                                                        //operator T&(),因此会隐式转换为Widget&类型（main中的w），因此起到
                                                        //的效果就好象main中的w直接被按引用传递到线程函数中来。

    cout << "thread end.(id = " << std::this_thread::get_id() << ")" << endl;
}

int main()
{
    //1. 向std::thread构造函数传参
    cout << "main thread begin...(id = "<<std::this_thread::get_id()<<")"<< endl;
    Widget w;
    cout << "-----------test std::thread constructor----------------------- "<< endl;
    //1.1 std::thread默认的按值传参方式：所有的实参都是被拷贝到std::thread对象的tuple中，即以副本形式被保存起来。
    std::thread t1(test_ctor, w); //注意，w是按值保存到std::thread中的，会调用其拷贝构造函数。
    t1.join();
    cout << "w.mutableInt = " << w.mutableInt << endl; //0，外部的w没受影响。mutableInf仍为0。

    cout << endl;

    //1.2 std::thread按引用传参(std::ref) 
    std::thread t2(test_ctor, std::ref(w)); //注意，w是按引用传入到std::thread中的，不会调用其拷贝构造函数。
    t2.join();
    cout << "w.mutableInt = " << w.mutableInt << endl; //1，由于w按引用传递，mutableInf被修改为1。

    cout << "------------------test thread function------------------------ " << endl;
    //2. 向线程函数传递参数
    //2.1 线程函数的参数为引用时
    //2.1.1 线程函数形参为T&
    //std::thread t3(updateWidget_ref, w); //编译失败，因为std::thread内部是以右值形式向线程函数updateWidget_ref(Widget&)传
                                           //参的，而右值无法用来初始化Widget&引用。
    std::thread t3(updateWidget_ref, std::ref(w)); //ok,原因类似test_ctor函数中的分析。即当线程函数的形参为T&时，
                                                   //一般以std::ref形式传入
    t3.join();
    //2.1.2 线程函数形参为const T&
    std::thread t4(updateWidget_cref, w); //ok，但要注意w会先被拷贝构造一次，以副本形式保存在thread中。该副本再被以右值
                                          //形式传递给线程函数updateWidget_cref(const Widget&)，而const T&可接受右值。
    t4.join();

    //2.2 隐式类型转换及临时对象
    const char* name = "Santa Claus";
    //注意：
    //（1）当向std::thread传入类成员函数时，必须用&才能转换为函数指针类型
    //（2）类成员函数的第1个参数是隐含的this指针，这里传入&w。
    //（3）本例会发生隐式类型转换，首先name在主线程中以const char*类型作为副本被保存在thread中，当向线程函数
    //     Widget::func(const string&)传参时，会先将之前的name副本隐式转换为string临时对象再传入，因此线程函数的形参中
    //     需要加const修饰。同时要注意，这个隐式转换发生在子线程调用时，即在子线程中创建这个临时对象。这就需要确保主线
    //     程的生命周期长于子线程，否则name副本就会变成野指针，从而无法正确构造出string对象。
    std::thread t5(&Widget::func, &w, name); //ok。
    t5.join();  //如果这里改成t5.detach,并且如果主线程生命期在这行结束时，就可能发生野指针现象。

    std::thread t6(&Widget::func, &w, string(name)); //为了避免上述的隐式转换可以带来的bug。可以在主线程先构造好这个
                                                     //string临时对象，再传入thread中。（如左）
    t6.join();

    //以下证明隐式转换发生在子线程中
    cout << endl;
    std::thread t7(updateWidget_implicit, 1); //会将1隐式转换为Widget,这个隐式转换发生在子线程。因为1会先以int型的副本
                                              //保存在t7中，当向线程函数传参时，才将int通过Widget的类型转换构造转成Widget。
    t7.join();

    cout << "main thread end.(id = " << std::this_thread::get_id() << ")" << endl;

    return 0;
}
/*输出结果：
main thread begin...(id = 8944)
Widget(), thread id = 8944
-----------test std::thread constructor-----------------------
Widget(const Widget& w), thread id = 8944 //w被按值保存std::thread中。会调用拷贝构造函数
thread begin...(id = 17328)
w.matableInt = 1       //只是修改std::thread中w副本的值。
thread end.(id = 17328)
w.mutableInt = 0       //main中的w没被修改

thread begin...(id = 5476)
w.matableInt = 1         //按std::ref传递既修改std::thread中w副本的值，也修改了main中w的值。
thread end.(id = 5476)
w.mutableInt = 1
------------------test thread function------------------------
invoke updateWidget_ref, thread id =17828
Widget(const Widget& w), thread id = 8944
invoke updateWidget_cref, thread id =2552
void func(string& s),  thread id = 11332
void func(string& s),  thread id = 17504

Widget(int i), thread id = 8996 //隐式转换发生在子线程8996中
invoke updateWidget_implicit, thread id =8996
main thread end.(id = 8944)
*/
```