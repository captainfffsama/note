#CPP 

[toc]

中文编码是一个复杂而繁琐的问题，在 C++ 程序设计中尤是如此。近期笔者在工作中对这一点颇有体会，故而在总结一些经验。

# 一、ASCII 码

[`ASCII 码`](https://ascii.cl/)，全称 American Standard Code for Information Interchange，一共规定了 128 个字符的编码，包括 10 个阿拉伯数字（从 `0x30` 开始）、26 个大写字母（从 `65` 开始）、26 个小写字母（从 `97` 开始），33 个英文标点符号，及 32 个控制字符，还有最后一个 `0x7F`。

128 个字符，至少需要 7 个比特（bit）来表示，而一个字节（byte）有 8 个比特，故将字节的最高位的比特规定为 0，使用剩下的 7 个比特，刚好可以完整的表示 ASCII 码规定的字符。

阿拉伯数字的编码从 `0x30` 到 `0x39`，按顺序分别表示 0 到 9 这 10 个字符。这样带来了一个优势：可以直接做字符的减法，得到字符对应的数字大小。大写字母和小写字母亦是如此。举个例子：

```cpp
char ch_9 = '9';
int value = ch_9 - '0';
assert(value == 9); 
```

大写字母从 `0x41` 开始，小写字母从 `0x61` 开始。注意观察，二者相差了一个 `0x20`，即 32。任意一个小写字母对比对应的大写字母，仅第 6 个比特有 1 与 0 的不同。进而可以通过这一点进行大小写字母的判断及其转换。

ASCII 中的控制字符较为少用，有印象的仅仅是 `Bell` 字符（0x07）。大一学习编程的时候发现可以通过 `printf("\a");` 使用电脑发出蜂鸣声，如今在 Mac 上尝试依然有效。

# 二、中文编码

ASCII 码仅规定了 128 个字符，只能满足英文的基本需求。一个字节最多能表示 256 个字符，而中文的常用汉字就有数千了，故而需要使用多个字节来表示汉字。两个字节可以表示的字符上限为 65536，绝大部分情况下能够满足汉字使用的需求了。经典的汉字编码包括 GBK、GB2312、GB18030、CP939 等。

在汉字编码中，之前 ASCII 码没有使用的最高位派上了用场。如果一个字节最高位是 0，说明这个字节便是 ASCII 码，查表解析即可；如果最高位非 0，那么说明这是一个多字节编码，需要联合下一个字节一起进行解析。

不同的编码，也就意味着要查询不同的表，也就会得到不同的解码结果。年纪大点的人应该会懂，把小说下载到 MP3 里结果都是乱码的痛苦。再加上这个世界不止有中文，全球各个地区的文字、符号数量远超出两个字节可以表示的范围，这时“统一度量衡”就显得尤为重要了。

# 三、Unicode

Unicode 便是便是文字和符号的统一度量衡。Unicode，Unique Code，Universe Code，全世界每一个字符对应一个唯一的编码。Unicode 收录了大量的汉字，汉字的编码从 `0x4E00` 开始，到 `0x9FFF` 结束。

然而 Unicode 仅仅定义了符号与二进制编码的关系，但没有定义如何存储和解析这些二进制编码。如果直接将二进制编码写入文件，那么读取时会产生歧义。例如 `4E 00 41`，你无法知道这记录的是 1 个字符，还是 2 个字符（可以解码为“一A”），或者是 3 个字符（可以解码为“N\[空\]A”）。如果统一每个字符的记录长度，那么对于常用中文便需要至少 3 个字节来表示一个符号，那么对于全英的文件则太浪费了。

# 四、UTF-8

Unicode 解决了编码统一的问题，但没有解决编码存储和解析的问题。UTF-8 则解决了 Unicode 没有解决的问题。

UTF-8 是一种变长编码，会使用 1 到 4 个字节表示一个字符，类似于哈夫曼树，具体与 Unicode 的映射关系如下（复制自参考文献1）：

| Unicode 范围（十六进制） | UTF-8 编码方式（二进制） |
| --- | --- |
| 0000 0000 ~ 0000 007F | 0xxxxxxx |
| 0000 0080 ~ 0000 07FF | 110xxxxx 10xxxxxx |
| 0000 0800 ~ 0000 FFFF | 1110xxxx 10xxxxxx 10xxxxxx |
| 0001 0000 ~ 0010 FFFF | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx |

UTF-8 对于原有的 ASCII 码完全兼容，而最高位的 1 的数量表示当前字符占用的字节数。可以通过上表将 Unicode 转换为 UTF-8 编码，即将 x 按照高低位顺序替换为 Unicode 的二进制位，缺了则使用 0 补齐。以汉字“一”为例，其 Unicode 编码为 `0x4E00`，对应的二进制为 `100 1110 0000 0000` ，二进制共计 15 位。填充到 `1110xxxx 10xxxxxx 10xxxxxx` 中，最高位缺了一位，使用 0 补齐，最终可得 `11100100 10111000 10000000`，即 `E4 B8 80`。使用一行 Python3 代码验证一下：

```cpp
print(b'\xe4\xb8\x80'.decode()) 
```

UTF-8 编码还保持着一个优秀的特性，无论是使用左对齐（字符串排序），还是右对齐（数值排序），UTF-8 编码始终保持着与 Unicode 一致的大小顺序。举个栗子🌰，字符串 u8"A" < u8"一", 同时宽字符 wchar\_t(L'A') < wchar\_t(L'一')。仔细想想还是蛮有意思的。

根据映射关系表可知，英文字符在 UTF-8 中使用一个字节存储，中文字符使用三个字节存储。可以根据映射关系轻松地写出 Unicode 与 UTF-8 间的转换程序。根据汉字在 Unicode 中的编码范围，可以计算所有汉字的 UTF-8 编码并输出。C++ 示例程序如下（C++ 14）：

```cpp
#include <iostream>
#include <fstream>
#include <memory>

std::unique_ptr<char[]> unicode_to_utf8(uint16_t unicode) {
  const int k_buffer_length = 3;
  auto buffer = std::make_unique<char[]>(k_buffer_length);
  buffer[0] = 0xE0 | (unicode >> 12);
  buffer[1] = 0x80 | ((unicode >> 6) & 0x3F);
  buffer[2] = 0x80 | (unicode & 0x3F);
  return buffer;
}

int main() {
  const auto k_chinese_encoding_begin = 0x4E00;
  const auto k_chinese_encoding_end = 0x9FFF;
  const auto k_output_filename = "test_01.txt";

  std::ofstream out(k_output_filename, std::ios::out | std::ios::binary);
  for (int i = k_chinese_encoding_begin; i <= k_chinese_encoding_end; i ++) {
    uint16_t unicode = i;
    auto buffer = unicode_to_utf8(unicode);
    out.write(buffer.get(), 3);
    out.write("\n", 1);
  }

  out.close();
} 
```

[点击此处查看完整的汉字 Unicode 对照表](https://sf-zhou.github.io/programming/chinese_unicode_encoding_table.htm)。

UTF-16 与 UTF-8 类似，使用变长编码。不同的是 UTF-16 使用一个或者两个 16bit 大小的编码单元，这样单个 Unicode 字符在 UTF-16 编码下字节长度为 2 或者 4。常用汉字在 UTF-16 中均可使用 2 个字节表示，但数字和字母也是 2 个字节。换句话说，如果是英文较多，使用 UTF-16 会产生较大的浪费；如果是中文较多，相较于 UTF-8 会节约不少空间。坏消息是 UTF-16 与 ASCII 码不兼容，应用上不如 UTF-8 广泛。

# 五、char 与 std::string

从上一节的代码中可以看到，直接在文件中写入 UTF-8 编码对应的三个 char 字节，即可输出文字。实际上程序输出的、文件记录的，都是字节流，类似于 Python3 中的 `bytes` 的概念。字节流本身并不清楚自己的编码方式，而打开文件的软件会根据字节流本身推测使用的编码。例如字节流 \[0xE4, 0xB8, 0x80\]，GBK 编码解析到 0x80 后无法找到后续的字节、解析失败，ASCII 无法解析 0xE4 同样解析失败，而 UTF-8 则可以很好地完成解码。

`char` 与 `std::string` 的功能与字节流一致。日常使用中可以通过 C++ 中的 `cout << "雾失楼台，月迷津渡" << endl;`，直接打印得到对应的汉字字符串，是因为代码文件编码和控制台编码显示的编码一致。以 Mac 为例，可以尝试将代码文件的编码切换为 CP939，vim 的话指令为 `:e ++enc=cp936`，再尝试编译和运行，就会得到编译器提出的 Warning 及输出的乱码。

然而直接使用字节流会产生一些问题，例如无法直接得到字符串的真实字符长度。UTF-8 编码的单个汉字长度为 3，英文长度为 1，中英文混合的字符串无法直接得到字符长度，需要先进行完整的解码。进而字节流也无法直接实现字符串的截取。例如：

```cpp
#include <iostream>
#include <string>
using std::cout;
using std::endl;
using std::string;

int main() {
  const string love_cpp = "我爱C++";
  cout << love_cpp.length() << endl;
  
} 
```

直到现在为止，C++ 中获取 UTF-8 编码的字符串字符长度都是一个麻烦的问题，一般需要依赖第三方库。这一点感觉可以借鉴 Python3，将字节流 `bytes` 和字符串 `str` 分割开，并提供方便的转换。

# 六、wchar_t 与 std::wstring

目前为止 C++ 没有采用上一节说的方案，而是大力出奇迹，提出使用 `wchar_t` 解决多字节编码带来的问题。既然多字节计算长度不准确，那就直接使用一个类型代替多字节。目前 UTF-8、Unicode 等编码至多需要 4 个字节，那就直接定义 `wchar_t` 为 4 个字节，所有问题都可以迎刃而解了。

是的，这不是开玩笑。`wchar_t` 具体定义为 2 个字节，还是 4 个字节并没有明确规定，视平台而定。在笔者的 Mac 上可以找到以下的定义：

```cpp
typedef int __darwin_wchar_t;
typedef __darwin_wchar_t wchar_t; 
```

而 `std::wstring` 则是 `string` 的 `wchar_t` 版，其定义为：

```cpp
typedef basic_string<wchar_t> wstring; 
```

使用宽字符类型可以解决字符串长度计算的问题，但会浪费了大量的存储空间。并且当前的 `std::wstring` 无法直接使用 `std::cout` 输出到标准输出流、无法使用 `std::ofstream` 输出到文件，而需要使用适配的 `std::wcout` 和 `std::wofstream`。综合来看，并不易用。

# 七、String Literal

C++ 代码中可以使用 "Hello World" 和 L"Hello World" 来声明字符串和宽字符串常量。C++ 11 开始支持 UTF-8、UTF-16 和 UTF-32 字符串常量的声明，分别使用 u8""、u"" 和 U"" 作为声明的标志，详细说明如下（复制自[参考文献2](https://en.cppreference.com/w/cpp/language/string_literal)）：

1.  Narrow multibyte string literal. The type of an unprefixed string literal is `const char[]`.
2.  Wide string literal. The type of a `L"..."` string literal is `const wchar_t[]`.
3.  UTF-8 encoded string literal. The type of a `u8"..."` string literal is `const char[]`.
4.  UTF-16 encoded string literal. The type of a `u"..."` string literal is `const char16_t[]`.
5.  UTF-32 encoded string literal. The type of a `U"..."` string literal is `const char32_t[]`.

#  八、最佳实践？
本段仅为个人经验，仅供参考。

IO 时统一使用 UTF-8，处理时统一用 Unicode。举个例子🌰：

```cpp
#include <codecvt>
#include <fstream>
#include <iostream>

int main() {
  const std::locale utf8( std::locale(), new std::codecvt_utf8<wchar_t> );
  const std::wstring love_cpp = L"\x6211\x7231""C++";  

  std::wcout.imbue(utf8);
  std::wcout << love_cpp << std::endl;  
  std::wcout << love_cpp.length() << std::endl;  
  std::wcout << love_cpp.substr(0, 2) << std::endl;  

  const auto k_output_filename = "test_02.txt";
  std::wofstream f_out(k_output_filename, std::ios::out);
  f_out.imbue(utf8);
  f_out << love_cpp << std::endl;
  f_out.close();

  std::wifstream f_in(k_output_filename, std::ios::in);
  f_in.imbue(utf8);
  std::wstring value;
  f_in >> value;
  std::wcout << (value == love_cpp ? "Yes": "Opps") << std::endl;  
  f_in.close();
} 
```

### [参考文献](#%E5%8F%82%E8%80%83%E6%96%87%E7%8C%AE)

1.  [阮一峰. "字符编码笔记：ASCII，Unicode 和 UTF-8."](http://www.ruanyifeng.com/blog/2007/10/ascii_unicode_and_utf-8.html)
2.  [C++ reference. "C++ Language String Literal".](https://en.cppreference.com/w/cpp/language/string_literal)
3.  [Wikipedia. "UTF-16".](https://en.wikipedia.org/wiki/UTF-16)