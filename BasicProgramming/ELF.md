---

mindmap-plugin: rich

---

# ELF
``` json
{"mindData":[[{"id":"1d12a171-a04b-7210","text":"ELF","isRoot":true,"main":true,"x":4000,"y":4000,"isExpand":true,"layout":{"layoutName":"mindmap2","direct":"mindmap"},"stroke":""},{"id":"4532a39d-33b8-c600","text":"代码段 text","stroke":"#100277","x":4127,"y":3946,"layout":{"layoutName":"mindmap2","direct":"mindmap"},"isExpand":true,"pid":"1d12a171-a04b-7210"},{"id":"cb7c6de5-fd2d-4951","text":"数据段 data","stroke":"#ffadae","x":4127,"y":4006,"layout":null,"isExpand":true,"pid":"1d12a171-a04b-7210"},{"id":"c8f9bb61-b1d1-d69f","text":"段标 section table","stroke":"#819316","x":4127,"y":4066,"note":"用来管理ELF中所有段的基本信息，想到与ELF文件的结构地图。编译器，链接器和装载器都依靠段表来定位和访问各个段属性。段表位置是ELF文件头中指明的。","layout":null,"isExpand":true,"pid":"1d12a171-a04b-7210"},{"id":"4308f964-6c58-a16e","text":"符号段 .symtab","stroke":"#a561cc","x":3786,"y":3976,"note":"符号就是用来标记一段代码或者数据，c 中写的变量名，函数名都是一种符号。\n\n而符号段就记录了这些函数名等符号的信息。\n\n","layout":null,"isExpand":true,"pid":"1d12a171-a04b-7210"},{"id":"9d41ac4e-5d2f-010f","text":"字符串段","stroke":"#f73dc2","x":3839,"y":4036,"note":"符号段中，每个符号信息使用固定字节数的结构，无法表示长度不同的字符串名称，因此将所有字符串放到一个段中，符号名通过记录符号名在字符串段中的偏移值即可拿到字符串","layout":null,"isExpand":true,"pid":"1d12a171-a04b-7210"}],[{"id":"6d6da6b8-fae9-ce8a","text":"freeNode","main":false,"x":10,"y":7,"layout":{"layoutName":"mindmap2","direct":"mindmap"},"isExpand":true,"stroke":""}]],"induceData":[],"wireFrameData":[],"relateLinkData":[],"calloutData":[{"nodeId":"1d12a171-a04b-7210","rootData":{"id":"c31386b1-9db4-e661","text":"参考","nodeType":"callout","style":{"background-color":"#f06","color":"#fff","font-size":"12px"},"x":4027,"y":3781,"note":"1. https://zhuanlan.zhihu.com/p/313161665\n2. https://zhuanlan.zhihu.com/p/311251561","layout":null,"isExpand":true,"stroke":"","point":{"x":4049,"y":4000},"box":{"dx":0.3103448275862069,"dy":-4.211538461538462,"px":0.5632183908045977,"py":0}},"color":"#f06","direct":"top"}]}
```
