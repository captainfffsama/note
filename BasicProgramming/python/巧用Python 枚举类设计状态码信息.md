#python  #待处理 


引言
--

> 在 `web` 项目中，我们经常使用自定义状态码来告知请求方请求结果以及请求状态；在 `Python` 中该如何设计自定义的状态码信息呢？

  

普通类加字典设计状态码
-----------

```python

class RETCODE:
    OK                  = "0"
    ERROR               = "-1"
    IMAGECODEERR        = "4001"
    THROTTLINGERR       = "4002"
    NECESSARYPARAMERR   = "4003"
    

err_msg = {
    RETCODE.OK                 : "成功",
    RETCODE.IMAGECODEERR       : "图形验证码错误",
    RETCODE.THROTTLINGERR      : "访问过于频繁",
    RETCODE.NECESSARYPARAMERR  : "缺少必传参数",
}

```

单独利用一个字典进行状态码信息对照，这样设计一旦状态码多了就不好对照，再使用过程中也没那么方便，简单试下组织一个成功的信息

```python
data = {
    'code': RETCODE.OK,
    'errmsg': err_msg[RETCODE.OK]
}
```

  

巧用枚举类设计状态码信息
------------

> 利用枚举类就可以巧妙的设计状态码信息

  

### 枚举类的定义

```python
from enum import Enum


class StatusCodeEnum(Enum):
    """状态码枚举类"""

    OK = (0, '成功')
    ERROR = (-1, '错误')
    SERVER_ERR = (500, '服务器异常')
```

普通的类继承 `enum` 模块中的 `Enum` 类就变成了枚举类。

  

### 枚举类的使用

在 `ipython` 中测试使用下

```python
In [21]: ok = StatusCodeEnum.OK

In [22]: type(ok)
Out[22]: <enum 'StatusCodeEnum'>

In [23]: error = StatusCodeEnum.ERROR

In [24]: type(error)
Out[24]: <enum 'StatusCodeEnum'>

In [26]: ok.name
Out[26]: 'OK'

In [27]: ok.value
Out[27]: (0, '成功')

In [28]: error.name
Out[28]: 'ERROR'

In [29]: error.value
Out[29]: (-1, '错误')
```

枚举类中的每一个属性都返回一个枚举对象，其中枚举对象有两个重要的属性 `name`, `value`

*   _**name**_ 枚举对象在枚举类中的属性名
*   _**value**_ 则是枚举对象在枚举类中对应属性名的值


用枚举类组组织一个成功的响应信息

```python
code = StatusCodeEnum.OK.value[0]
errmsg = StatusCodeEnum.OK.value[1]
data = {
    'code': code,
    'errmsg': errmsg
}
```

咋一看虽然状态码信息一一对照了，也很简洁，但使用起来还是有点麻烦，还有一点就是

`StatusCodeEnum.OK.value[0]` 这样的语法不能立马见名知义。因此还需对枚举类进行封装

  

### 封装枚举类

```python
from enum import Enum


class StatusCodeEnum(Enum):
    """状态码枚举类"""

    OK = (0, '成功')
    ERROR = (-1, '错误')
    SERVER_ERR = (500, '服务器异常')

    @property
    def code(self):
        """获取状态码"""
        return self.value[0]

    @property
    def errmsg(self):
        """获取状态码信息"""
        return self.value[1]
```

通过 `@property` 装饰器把类型的方法当属性使用，由于 **枚举类.属性名** 对应着不同的枚举对象就很好的把状态码和信息进行了封装。看看外部调用的结果

```python
In [32]: StatusCodeEnum.OK.code
Out[32]: 0

In [33]: StatusCodeEnum.OK.errmsg
Out[33]: '成功'

In [34]: StatusCodeEnum.ERROR.code
Out[34]: -1

In [35]: StatusCodeEnum.ERROR.errmsg
Out[35]: '错误'
```

具体 `@property` 装饰器的使用详解，可以移步到 [Python中property的使用技巧](https://juejin.cn/post/6959143711699632142 "https://juejin.cn/post/6959143711699632142")

继续模拟组织响应数据

```python
data = {
    'code': StatusCodeEnum.OK.code,
    'errmsg': StatusCodeEnum.OK.errmsg
}
```

这下终于可以接受了。

  

状态码信息枚举类
--------

> 分享一波我平时用的状态码信息枚举类，供大家参考参考。

```python
from enum import Enum


class StatusCodeEnum(Enum):
    """状态码枚举类"""

    OK = (0, '成功')
    ERROR = (-1, '错误')
    SERVER_ERR = (500, '服务器异常')

    IMAGE_CODE_ERR = (4001, '图形验证码错误')
    THROTTLING_ERR = (4002, '访问过于频繁')
    NECESSARY_PARAM_ERR = (4003, '缺少必传参数')
    USER_ERR = (4004, '用户名错误')
    PWD_ERR = (4005, '密码错误')
    CPWD_ERR = (4006, '密码不一致')
    MOBILE_ERR = (4007, '手机号错误')
    SMS_CODE_ERR = (4008, '短信验证码有误')
    ALLOW_ERR = (4009, '未勾选协议')
    SESSION_ERR = (4010, '用户未登录')

    DB_ERR = (5000, '数据错误')
    EMAIL_ERR = (5001, '邮箱错误')
    TEL_ERR = (5002, '固定电话错误')
    NODATA_ERR = (5003, '无数据')
    NEW_PWD_ERR = (5004, '新密码错误')
    OPENID_ERR = (5005, '无效的openid')
    PARAM_ERR = (5006, '参数错误')
    STOCK_ERR = (5007, '库存不足')

    @property
    def code(self):
        """获取状态码"""
        return self.value[0]

    @property
    def errmsg(self):
        """获取状态码信息"""
        return self.value[1]
```

  
