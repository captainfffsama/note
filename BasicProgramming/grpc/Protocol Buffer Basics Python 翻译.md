#protobuff 

[原文](https://developers.google.com/protocol-buffers/docs/pythontutorial)  

[toc]

本教程将介绍如何使用 python 来处理 protobuff .通过创建一个简单的应用示例,您将学会:  

- 使用`.proto`文件来定义消息格式.
- 使用 protocol buff 编译器
- 使用 python API 来读写消息.  

本教程不是 python 处理 protobuff 的完整指南.更多信息参见: [Protocol Buffer Language Guide](https://developers.google.com/protocol-buffers/docs/proto), [Python API Reference](https://googleapis.dev/python/protobuf/latest/),[Python Generated Code Guide](https://developers.google.com/protocol-buffers/docs/reference/python-generated),[Encoding Reference](https://developers.google.com/protocol-buffers/docs/encoding)  

# 为何使用 Protocol Buffers?
我们将创建一个类似"地址簿"的应用,它将从文件中读取联系人信息,这些信息包括人名,ID, 电子邮件地址和电话.   
您会如何来序列化和检索这种结构数据呢? 有以下几个方式来解决这个问题: