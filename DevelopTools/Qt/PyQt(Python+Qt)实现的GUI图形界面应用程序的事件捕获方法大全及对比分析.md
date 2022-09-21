#python 
#qt

[toc]

# 一、 概述

PyQt的图形界面应用中，事件处理类似于Windows系统的消息处理。一个带图形界面的应用程序启动后，事件处理就是应用的主循环，事件处理负责接收事件、分发事件、接收应用处理事件的返回结果，在程序中捕获应用关注的事件触发相关事件处理是良好UI开发的必经之路。那么在PyQt的图形界面应用中，有哪些方法可以捕获事件以进行处理呢？下面我们就来分析一下。

# 二、 应用层级的事件捕获

## 2.1、notify方法捕获应用事件
PyQt的事件处理是从应用主程序开始的，在PyQt应用主程序中，真正负责事件分发处理的是QApplication类的notify方法(或称为notify函数），该方法负责向接收者发送事件，返回接收事件对象的处理程序返回的值。因此要在应用中捕获事件并进行处理，只要通过从QApplication类派生自定义的应用类并重写notify方法就可以捕获应用接收到的所有事件。

### 2.1.1、notify的语法
notify(QObject receiver, QEvent event）

**其中：**   
1、参数receiver表示将事件发送给谁；  
2、event就是事件参数，类型为QEvent ，如果不了解请参考《[PyQt学习随笔：Qt事件类QEvent详解](https://blog.csdn.net/LaoYuanPython/article/details/102527965)》；  
3、返回值为receiver的事件处理方法的返回值，如果返回值是False表示接收者对事件不感兴趣，需要应用将事件信息继续向下传到接收者的父级，依此类推，直至顶级对象，如果返回True表示消费了事件，事件不会再往下传递。

### 2.1.2、一段notify重写的示例代码

```python
class App(QApplication):
    def notify(self, eventobject: QObject, event: QEvent):
        """
        本次重写notify是为了截获应用的所有事件，并针对鼠标和键盘按下事件输出事件相关的信息
        :param eventobject: 事件接收对象
        :param event: 具体事件
        :return: True表示事件已经处理，False表示没有处理，需要继续往下传递
        """

        eventtype = event.type()
        flag = False
        if eventtype==QEvent.Close  or eventtype==QEvent.KeyPress or eventtype == QEvent.MouseButtonPress: 
            flag=True
            
                
            
        if flag:
            print(f"In app notify:事件类型值={eventtype}，事件接收者:{eventobject},parent={eventobject.parent()},child={eventobject.children()}")

        ret = super().notify(eventobject, event)
        if flag:
            print(f"App notify end,事件接收者:{eventobject}，事件返回值={ret},app={self},parent={eventobject.parent()}")
        return ret

```

### 2.1.3、重写notify方法后的应用主程序示例代码
由于重写notify方法需要使用从QApplication派生自定义类，因此应用主程序的应用对象应该从新派生类构建，实例代码如下：

```python
if __name__ == '__main__':
        app = App(sys.argv)
        w = eventCap()  
        w.show()
        sys.exit(app.exec_())

```

## 2.2、安装应用级的事件过滤方法
### 2.2.1、概述 
要捕获应用级的事件，除了Notify方法外，还可以采用安装应用级的事件过滤方法。  
事件过滤会接收到所有发给该对象的所有事件，事件过滤可以终止事件或继续将事件提交到这个对象往下处理。事件过滤通过对象的eventFilter() 方法来接收事件，如果事件需要被终止，则eventFilter()方法需要返回True，否则返回False。  
一个对象上可以安装多个事件过滤，这时候最后安装的事件过滤在事件到达时最先处理。

### 2.2.2、安装应用级事件过滤的步骤 
要安装应用级的事件，需要如下步骤：  
1、 在某个用来进行事件监控的从QObject派生的自定义类中重写派生类的eventFilter方法；  
2、 在需要监控的对象上调用installEventFilter安装事件监控，由于本部分介绍的是应用级的事件过滤，因此需要使用应用的实例对象来安装。

### 2.2.3、eventFilter方法的语法
bool eventFilter(QObject watched, QEvent event)  
**其中：**   
1、watched：监视对象，就是被安装了eventFilter的对象；  
2、event：接收到的事件信息；  
3、返回值：为True表示事件到此结束，即该事件不会再往下传递，否则会继续传递。

### 2.2.4、installEventFilter方法的语法  
monitorObj.installEventFilter(QObject filterObj)  
**其中：**   
1、 monitorObj：需要进行事件刷选的对象；  
2、 filterObj：重写了eventFilter方法的对象；  
3、 该方法无返回值。  
注意：monitorObj和filterObj在多线程应用中，这两个对象必需在同一个线程内，否则installEventFilter不起作用。

### 2.2.5、自定义事件刷选类代码示例

```python
class eventMonitor(QObject):
    def eventFilter(self, objwatched, event):

            eventType = event.type()

            flag =  eventType == QEvent.MouseButtonPress or eventType == QEvent.KeyPress or eventType == QEvent.Close 
            if flag:
                print(f"In eventMonitor eventFilter:事件类型值={eventType}，事件objwatched={objwatched},parent={objwatched.parent()},child={objwatched.children()}")

            ret = super().eventFilter(objwatched, event)
            if flag: self.log(f'eventMonitor eventFilter end,ret={ret}')
            return ret

```

### 2.2.6、事件过滤安装代码示例
下面这段代码对应用和应用窗口的一个按钮安装了同一个事件刷选对象：

```python
if __name__ == '__main__':
        app = QApplication(sys.argv)
        w = eventCap() 
        w.show()
        monitorObj = eventMonitor() 
        app.installEventFilter(monitorObj) 
        w.pushButton_eventtest.installEventFilter(monitorObj) 
        sys.exit(app.exec_())

```

# 三、 部件级的事件捕获方法

## 3.1、基于事件刷选进行部件级的事件捕获
对部件使用事件刷选就可以实现部件级的事件捕获，相关方法与应用级的事件刷选完全一样，只是在调用installEventFilter安装事件时，调用的对象由应用改成了对应部件对象。在2.2.6部分介绍的案例就同时安装了一个应用级的事件刷选和一个部件级的事件刷选。在此不再重复介绍。

## 3.2、重写部件类的event方法捕获对象的事件  
### 3.2.1、概述 
在PyQt的部件对象中，都有从QWidget中继承的方法event，而QWidget.event是对QObject类定义的虚拟方法event的实现。在部件类中，event方法是处理部件收到的所有消息，因此如果部件类是从QWidget等PyQt提供的部件类派生的自定义类，则可以在自定义类中重写event方法实现部件收到的所有事件的处理。

### 3.2.2、event方法的语法  
bool event（QEvent e）

**其中：**   
1、 参数e：为事件；  
2、 返回值：如果事件被识别并处理应该返回True，对于没有被应用识别和处理的事件，需要调用父类的event方法以保证事件的正确处理，此时应该返回父类event方法的返回值。

### 3.2.3、注意 
1、该方法中只能捕获确认是发给对应对象的事件，不能捕获通过该对象转发给上级的事件；  
2、通过重写该方法可以捕获对象的所有事件，但Qt并不推荐这种使用方法，而应该通过重写具体事件的具体方法来捕获特定事件；  
3、event和特定事件的事件处理方法针对一个特定事件处理时，先调用event再调用特定事件的事件处理方法;  
4、如果event处理事件时，没有调用父类方法，则事件处理终止，对应的事件不能再被该事件的特定事件处理方法捕获；  
5、键盘按下和释放事件的处理方式与其他事件不同，event（）会检查键盘事件是否为tab和shift+tab释放事件，如果是尝试移动焦点。如果没有要将焦点移动到的小部件（或按键不是tab或shift+tab），event（）调用keyPressEvent（）处理该键盘按键事件。

### 3.2.4、示例代码

```python
 def event(self, eventobj):
            eventtype = eventobj.type()
            flag = False
            if eventtype == QEvent.Close or eventtype == QEvent.MouseButtonPress or  eventtype == QEvent.KeyPress: 
                flag = True
            if flag:
                self.log(f"In event，事件类型值={eventtype}，事件接收者:{self},parent={self.parent()},child={self.children()}")
            ret = super().event(eventobj)

            if flag:
                self.log(f"Event end,事件返回值={ret}")

            return ret

```

## 3.3、重写特定事件处理方法捕获对象的特定事件  
### 3.3.1、概述
大多数时候，我们无需截获所有事件进行处理，只需要进行特定事件的捕获和处理，当然可以用前面几种方法加上事件类型判断来进行捕获和处理，但这些方法会对应用的整体事件处理产生性能影响，因此最好是需要处理什么事件就捕获什么事件。这种情况建议通过自定义类重写部件的特定事件处理函数来实现。

### 3.3.2、常用特定事件列表  
- keyPressEvent： 键盘按下事件  
- keyReleaseEvent： 键盘释放事件  
- mouseDoubleClickEvent： 鼠标双击事件  
- mouseMoveEvent： 鼠标移动事件  
- mousePressEvent： 鼠标按下事件  
- mouseReleaseEvent： 鼠标释放事件  
- timerEvent： 定时器事件  
- dragEnterEvent： 拖拽进入当前窗口事件  
- dragLeaveEvent： 拖拽离开当前窗口事件  
- dragMoveEvent： 拖拽移动事件  
- enterEvent： 进入窗口区域事件  
- leaveEvent： 离开窗口区域事件  
- closeEvent： 关闭窗口事件  
- paintEven：界面绘制事件

### 3.3.3、示例代码

```python
    def keyPressEvent(self, keyevent):
        print(f"In keyPressEvent:键盘按键 {keyevent.text()},0X{keyevent.key():X} 被按下")


    def mousePressEvent(self, mouseEvent):
        print(f"In mousePressEvent:鼠标按下")

```

## 3.4、通过信号与槽函数机制捕获事件
信号与槽函数机制严格意义上来说已经不属于事件处理机制，但大多数部件的信号就是从部件常用的事件处理中产生的，利用信号连接一个应用实现的槽函数，就能实现特定事件的应用响应，因此也可以认为是用来捕获事件的一种机制。信号和槽函数是Qt最重要的机制之一，通过这种机制实现了界面和应用处理逻辑的分离，关于Qt信号和槽函数的内容可以查阅的资料很多，在此就不展开介绍。

# 四、 PyQt事件捕获几种方法的处理过程


## 4.1、事件处理流程
通过在一个应用中实现上面介绍的六种方法（两种应用级、四种部件级）来捕获事件，且不终止事件传递的情况下，会发现事件在这些方法的流转过程如下：  
![](https://img-blog.csdnimg.cn/20191017214842557.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xhb1l1YW5QeXRob24=,size_16,color_FFFFFF,t_70)
  
正常情况下，事件到达应用后，应用调用notify通知QWindow隐形窗口对象（关于该隐形窗口对象，请参阅《[PyQt学习遇到的问题：重写notify发送的消息为什么首先给了一个QWindow对象？](https://blog.csdn.net/LaoYuanPython/article/details/102582800)》后，再通知对应部件，部件收到后会进行事件处理，判断是否该接受该事件，如果接受了则事件处理终止，如果不接受则传给部件的父对象进行处理。

## 4.2、事件处理案例  
### 4.2.1 案例背景
在一个名为app的应用中，有个名为w的主窗口，主窗口上有个名为testButton的按钮，应用已经启动。下面案例的跟踪信息是使用上面示例代码输出的信息，相关界面如下：  
![](https://img-blog.csdnimg.cn/20191018141807342.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xhb1l1YW5QeXRob24=,size_16,color_FFFFFF,t_70)
  
注意：上面界面窗口显示的跟踪信息没有包含eventFilter方法的，详细输出信息需要看程序的打印输出。

### 4.2.2 案例1：使用鼠标点击testButton的按钮的事件处理过程
截获的事件及解释如下：

```python
1、	In app notify:事件类型值=2，事件接收者:<PyQt5.QtGui.QWindow object at 0x0000000004B17318>,parent=None,child=[]

```

**事件说明**：应用notify通知QWindow隐形窗口对象，事件类型值=2表示鼠标按键事件，事件类型取值及含义具体可参考《[PyQt学习随笔：Qt事件QEvent.type类型常量及其含义资料汇总详细内容速查](https://blog.csdn.net/LaoYuanPython/article/details/102527651)》;

```python
2、In eventMonitor eventFilter:事件类型值=2，事件objwatched=<PyQt5.QtGui.QWindow object at 0x0000000004B17318>,parent=None,child=[]
3、eventMonitor eventFilter end,ret=False

```

**事件说明**：应用的事件刷选捕获到发给QWindow隐形窗口对象的鼠标按键事件

```python
4、In app notify:事件类型值=2，事件接收者:<PyQt5.QtWidgets.QPushButton object at 0x0000000004638F78>,parent=<__main__.eventCap object at 0x0000000004638DC8>,child=[]

```

**事件说明**：应用notify通知QPushButton对象的鼠标按键事件

```python
5、In eventMonitor eventFilter:事件类型值=2，事件objwatched=<PyQt5.QtWidgets.QPushButton object at 0x0000000004638F78>,parent=<__main__.eventCap object at 0x0000000004638DC8>,child=[]
6、eventMonitor eventFilter end,ret=False

```

**事件说明**：应用的事件刷选捕获到发给QPushButton对象的鼠标按键事件

```python
7、In eventMonitor eventFilter:事件类型值=2，事件objwatched=<PyQt5.QtWidgets.QPushButton object at 0x0000000004638F78>,parent=<__main__.eventCap object at 0x0000000004638DC8>,child=[]
8、eventMonitor eventFilter end,ret=False

```

**事件说明**：主窗口部件的事件刷选捕获到发给QPushButton对象的鼠标按键事件

```python
9、App notify end,事件接收者:<PyQt5.QtWidgets.QPushButton object at 0x0000000004638F78>，事件返回值=True,app=<__main__.App object at 0x0000000004638E58>,parent=<__main__.eventCap object at 0x0000000004638DC8>
10、App notify end,事件接收者:<PyQt5.QtGui.QWindow object at 0x0000000004B17318>，事件返回值=True,app=<__main__.App object at 0x0000000004638E58>,parent=None

```

**事件说明**：应用的两次notify调用结束返回

```python
12、	In genevent:接收到信号,按钮<PyQt5.QtWidgets.QPushButton object at 0x0000000004638F78>被按下！

```

**事件说明**：槽函数genevent接收到QPushButton对象的鼠标按键事件

**4.2.3 案例2：焦点选中testButton后，按下键盘按键‘a’事件处理过程**  
截获的事件及解释如下：

```python
1、In app notify:事件类型值=6，事件接收者:<PyQt5.QtGui.QWindow object at 0x0000000004B17318>,parent=None,child=[]

```

**事件说明**：应用notify通知QWindow隐形窗口对象，事件类型值=6表示键盘按键事件

```python
2、In eventMonitor eventFilter:事件类型值=6，事件objwatched=<PyQt5.QtGui.QWindow object at 0x0000000004B17318>,parent=None,child=[]
3、eventMonitor eventFilter end,ret=False

```

**事件说明**：应用的事件刷选捕获到发给QWindow隐形窗口对象的键盘按键事件

```python
4、In app notify:事件类型值=6，事件接收者:<PyQt5.QtWidgets.QPushButton object at 0x0000000004638EE8>,parent=<__main__.eventCap object at 0x0000000004638DC8>,child=[]

```

**事件说明**：应用notify通知QPushButton对象的键盘按键事件

```python
5、In eventMonitor eventFilter:事件类型值=6，事件objwatched=<PyQt5.QtWidgets.QPushButton object at 0x0000000004638EE8>,parent=<__main__.eventCap object at 0x0000000004638DC8>,child=[]
6、eventMonitor eventFilter end,ret=False

```

**事件说明**：应用的事件刷选捕获到发给QPushButton对象的键盘按键事件

```python
7、In eventMonitor eventFilter:事件类型值=6，事件objwatched=<__main__.eventCap object at 0x0000000004638DC8>,parent=None,child=[<PyQt5.QtWidgets.QPushButton object at 0x0000000004638EE8>, <PyQt5.QtWidgets.QPushButton object at 0x0000000004638F78>, <PyQt5.QtWidgets.QLineEdit object at 0x0000000004B17048>, <PyQt5.QtWidgets.QWidget object at 0x0000000004B170D8>]
8、eventMonitor eventFilter end,ret=False

```

**事件说明**：主窗口部件的事件刷选捕获到发给QPushButton对象的键盘按键事件

```python
9、In event，事件类型值=6，事件接收者:<__main__.eventCap object at 0x0000000004638DC8>,parent=None,child=[<PyQt5.QtWidgets.QPushButton object at 0x0000000004638EE8>, <PyQt5.QtWidgets.QPushButton object at 0x0000000004638F78>, <PyQt5.QtWidgets.QLineEdit object at 0x0000000004B17048>, <PyQt5.QtWidgets.QWidget object at 0x0000000004B170D8>]

```

**事件说明**：主窗口部件的event方法捕获到键盘按键事件，但处理未完成

```python
10、In keyPressEvent:键盘按键 a,0X41 被按下

```

**事件说明**：主窗口部件的keyPressEvent方法捕获到键盘按键事件，处理完成

```python
11、Event end,事件返回值=True

```

**事件说明**：主窗口部件的event方法结束

```python
12、App notify end,事件接收者:<PyQt5.QtWidgets.QPushButton object at 0x0000000004638EE8>，事件返回值=True,app=<__main__.App object at 0x0000000004638E58>,parent=<__main__.eventCap object at 0x0000000004638DC8>
13、App notify end,事件接收者:<PyQt5.QtGui.QWindow object at 0x0000000004B17318>，事件返回值=True,app=<__main__.App object at 0x0000000004638E58>,parent=None

```

**事件说明**：应用的两次notify调用结束返回

**从上面两个案例对比来说：**   
1、 步骤1-6基本相同，只是事件类型不同；  
2、 第7-8步中，案例1由于鼠标按键事件被按钮接收了，所以窗口部件的事件刷选捕获到的是按钮的鼠标按键事件，案例2由于键盘事件按钮没有被接受，被往下传递给了其父节点主窗口，因此主窗口的事件刷选捕获到的是发给主窗口的键盘按键事件；  
3、 案例1中由于按钮接受了鼠标按键事件，而按钮没有重写event及mousePressEvent方法，因此整个事件在步骤7-8后就进入结束了，而案例2的事件在步骤7-8传递给了主窗口，因此触发了后续步骤的event方法以及后续的keyPressEvent方法。

# 五、 PyQt事件捕获方法的对比


![](https://img-blog.csdnimg.cn/20191017215657642.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xhb1l1YW5QeXRob24=,size_16,color_FFFFFF,t_70)

**注：本文示例案例的完整代码请到《[PyQt图形应用事件捕获案例.rar](https://download.csdn.net/download/laoyuanpython/11874964)》下载。** 



**博客地址：[_https://blog.csdn.net/LaoYuanPython_](https://blog.csdn.net/LaoYuanPython)**

