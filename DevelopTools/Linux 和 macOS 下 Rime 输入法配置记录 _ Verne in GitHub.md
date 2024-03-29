#Linux 

Rime IME 是中州韻輸入法引擎 (Rime Input Method Engine) 的缩写，项目网址如下：

*   [https://github.com/rime](https://github.com/rime)

按照官网给出的定义：Rime 不是一种输入法，是从各种常见键盘输入法中提炼出来的抽象的输入算法框架，这一套框架提供的灵活扩展性使得其不仅可以支持全拼，双拼等等中文输入，还可以扩展词库进行任何语言的输入，Rime 涵盖了大多数输入法的共性，所以在不同设定下，Rime 可化身为不同的输入法用来打字。Rime 是跨平台的输入法软件，Rime 输入方案可通用于以下发行版：

*   【中州韻】 fcitx-rime → Linux, 配置地址 `~/.config/fcitx/rime/`
*   【小狼毫】 Weasel → Windows
*   【鼠鬚管】 Squirrel → Mac OS X, 配置地址： `~/Library/Rime/`

很多东西看官方的 wiki 就能看明白，希望在你继续看下去之前，先看完下面几个网址：

*   [Rime 官方说明书](https://github.com/rime/home/wiki/UserGuide)
*   [Rime 官方定制指南](https://github.com/rime/home/wiki/CustomizationGuide)

相信如果你看完了官方的文档，应该对 Rime 的安装和配置有了一定的了解，相信你在看得过程中也已经把 Rime 对应的版本安装上了，不同发行版的配置目录不同，通过 `yaml` 文件来配置，本文就在官方文档的基础上调整了一些配置来适应我的习惯。你不一定要完全按照我的配置来，不过我相信读完这篇文章，你一定能够随心配置出适合你自己输入习惯的 Rime。

下面的文章不是一篇入门的文章，你可能需要自行先阅读官网提供的 Wiki，以及自行安装上 Rime 体验一下之后，再阅读后面的内容。

关于安装本来不想多说什么，基本上都是非常简单的，但是在 Linux 上的一些使用经验告诉我，Linux 发行版上的 Rime，fcitx-rime 要比 ibus 版本的要好用，虽然官方建议的是 ibus 版本，但是我还是建议你使用一下 fcitx 版本的，并且结合 fcitx 的扩展会开启一个新的世界，比如自带粘贴版历史（`Ctrl+;`)，自带日韩语输入，这一切会让使用体验上升一个高度。

```
sudo apt install -y fcitx fcitx-rime 
```

更多发行版见[官网](https://rime.im/download/)

在 macOS 上直接通过 Homebrew 安装：

```
brew cask install squirrel 
```

或者从源码自行编译安装：

*   [https://github.com/rime/squirrel/blob/master/INSTALL.md](https://github.com/rime/squirrel/blob/master/INSTALL.md)

在初了解 Rime 的时候经常会被 Rime 中的几个输入方案的名字搞混，其实这三个方案 `朙月拼音`，以及该方案的简体字版本，语句流版本都是同一个方案，不过在体验上略有差别，「语句流」风格的输入方案，在空格确认后，字词并不立即上屏，而是在输入句末的标点或者按下回车时整个句子才上屏。

安装完之后， `fcitx-rime` 的大部分的配置文件在 `~/.config/fcitx/rime` 下，如果是 `ibus` 版本，将对应的 fcitx 替换成 ibus 即是配置地址，如果打开该目录能看到一系列默认配置：

*   `default.yaml`: 全局配置
*   `weasel.yaml` : 发行版配置，比如 Windows 下就是 weasel, macOS 下面就是 `squirrel.yaml`
*   `installation.yaml`: 安装信息，包括了输入法的前端发行版，版本，安装时间，安装的ID，rime 的版本，以及最重要的输入法数据备份路径等
*   `*.schema.yaml`: 各输入方案的配置文件，一般认为 schema 前的是输入方案的名字，一个输入方案可能对应多个字典
*   `user.yaml`: 用户状态

用户自定义：

*   `*.dict.yaml` : 用户字典
*   `*.custom.yaml`: 用户对 `default.yaml`, `*.schema.yaml` 等配置文件进行修改的配置文件

修改 Rime 配置文件并使之生效的方法很简单，保存文件，在 Rime 菜单中点击**部署**，就可以立即看到效果。Rime 建议使用 `*.custom.yaml` 的方式来自定义配置，因为 Rime 升级时会覆盖 Rime 自己的默认配置文件。

如果配置了同步目录，那么在同步目录能看到

*   `*.userdb.txt` : Rime 会自动在一定时间后将用户的输入习惯生成快照文件，记录在 `userdb.txt` 文件中，分别记录了该输入方案下用户输入的历史纪录，备份级别：**重要**，具体文件内容解析可以参考后文
*   `UUID/` : 用户配置同步目录

虽然现在很多手机上的输入法支持双语输入，但是桌面版的输入法除了搜狗，其他真的做的不行，但是 Rime 支持很多种双语切换时的处理方式。比如某些情况下一直在输入中文，但是中间要输入英文，通常的做法就是按下 Shift 来快速切换到英文模式，那么这个时候，如何处理已经输入的内容，Rime 提供了如下的配置：

*   `inline_ascii`: 在输入法的临时英文编辑区内输入字母、数字、符号、空格等，回车上屏后自动复位到中文
*   `commit_text`: 已输入的**候选文字**上屏并切换至西文输入模式
*   `commit_code`: 已输入的**编码字符**上屏并切换至西文输入模式
*   `clear`: 丢弃已输入的内容并切换至西文输入模式
*   `noop`: 屏蔽该切换键

具体设置如下：

```
ascii_composer:
  good_old_caps_lock: true
  switch_key:
    Caps_Lock: noop
    Eisu_toggle: clear
    Shift_L: commit_code
    Shift_R: commit_text
    Control_L: commit_text
    Control_R: noop 
```

在这样的配置下，比如我已经输入了 `vim`，但是输入法并没有 `vim` 的候选词，这个时候我按下左 `Shift`，Rime 会按照配置，执行 `commit_code`，也就是将输入的 `vim` 上屏，并切换到英文输入模式。这是我一直比较习惯的用法。

在设置自定义键的时候左右 `Shift` 和 `fcitx` 的快捷键有冲突。`fcitx` 设置中 `额外的激活输入法快捷键` 的 `双侧 Shift` 改掉或者禁用。

对于左 `Shift` 是将候选词上屏，`Enter` 则是将输入的内容原封不动上屏，和我之前的习惯保持一致。

要扩展 Rime 的词库，就必须要先知道 Rime 中词库的分类，Rime 中词库有两大类，一类是 `*.dict.yaml` 文件编译后生成的 `*.table.bin` 文件，这一类被称为固态词典，另一大类是用户输入习惯词典，一般保存在 `*.userdb` 文件夹中。固态词典不会随着用户的输入而发生变化，用户可以通过更改 dict.yaml 文件，然后重新部署生效，而**用户词典**则会随着用户的输入而发生变化，从而影响候选词的位置。用户词库会在同步时合并到 `*.userdb.txt` 文件中，并同步到配置的同步文件夹中，从而做到跨设备的同步。

Rime 自带的联想词库有其自身的局限，但是 Rime 可以支持扩展词典，在 Rime 配置目录下新建 `*.dict.yaml` 这样的文件：

```
luna_pinyin.mywords.dict.yaml 
```

在文件中输入

```
# Rime dictionary
# encoding: utf-8

---
name: luna_pinyin.mywords
version: "2019.08.23"
sort: by_weight
use_preset_vocabulary: true
# 從 luna_pinyin.dict.yaml 導入包含單字的碼表
import_tables:
  - luna_pinyin
...

# table begins

# 自定義的詞條
中州韵	zhong zhou yun	1
小狼毫	xiao lang hao
自动注音的词 
```

这里需要注意的是 Rime 的码表，是 Tab 分割的三列，分别是文字、编码、使用频次，编码的定义以音节加空格分割。在制作自己的码表时一定要注意使用 Tab 分割。

编辑文件 `luna_pinyin.custom.yaml`

```
patch:
  translator/dictionary: luna_pinyin.mywords 
```

部署，就可以快速导入到 Rime 中。

*   [https://github.com/xiaoTaoist/rime-dict](https://github.com/xiaoTaoist/rime-dict)
*   官方提供的一個詞庫 [https://github.com/rime-aca/dictionaries](https://github.com/rime-aca/dictionaries)

部署之后尝试输入词库中任意一个比较复杂的词，理论上应该看到正确的词出现，否则可能词库未加载成功，那么需要到 `/tmp/rime*` 目录下查看 ERROR 日志。

关于如何制作自己的词库，后来我又写了一篇文章，可以参考[这里](https://einverne.github.io/post/2019/08/make-rime-dict.html)

在最开始的时候我最没有明白的一个配置就是 Rime 的同步机制，后来发现在 `installation.yaml` 文件中配置：

```
 sync_dic: "/path/to/rsync"
 installation_id: "mint-config" 
```

然后点击 Rime 图标，部署，同步，这样用户配置和词库都会在配置的同步路径中。如果不修改 `installation.yaml` 配置，那么 Rime 默认会生成一个 UUID 的目录来存放同步文件

如果更换了电脑，将同步文件同步下来，然后配置 `installation.yaml` ，执行部署，同步，部署，然后配置、用户词库都可以了。

不同电脑之间的同步内容会以 `installation_id` 为名生成一个文件夹，文件的配置不会相互互通，但是用户字典是会同步的。

简单地来描述 Rime 的同步机制就是，不同电脑都会在同步目录中新建一个自己的 UUID 或者指定名字的目录，该目录下会保存所有自定义的修改，包括用户输入习惯的字典 `*.userdb.txt` .

在点击同步时，Rime 会，会把同步目录中其他的 `userdb.txt` 合并：

```
sync/*/*.userdb.txt = 合并到 => 本地 *.userdb = 导出为 => sync/<installation_id>/*.userdb.txt`
其他电脑 = 複製來或從網盤同步下載 => sync/<installation_id>/ = 本地同步 => sync/<installation_id>/ = 複製走或同步上傳到網盤 => 其他电脑同步 
```

也就不需要手动去合并字典，只需要在合适的时机同步一下目录即可。Rime 会自动处理 `userdb.txt` 的合并问题。

不过需要注意的是 Rime 虽然会将自定义配置及词库同步到目录，但这只是一个单向的同步，其他机器依然需要手动处理自定义配置及词库。

在同步后可以在同步目录观察到 `*.userdb.txt` 每一种输入方案都会对应一个这样的文件，其中保存的是用户的输入历史，打开文件看很容易可以猜测到每一行中的含义：

```
# Rime user dictionary
#@/db_name	luna_pinyin.userdb
#@/db_type	userdb
#@/rime_version	1.2.9
#@/tick	425369
#@/user_id	mint-config
a 	吖	c=14 d=4.71446e-09 t=425369
a 	呵	c=1 d=6.28595e-10 t=425369
a 	啊	c=8781 d=3.80755 t=425369 
```

解释：

*   `c` 输入法 commit 的次数 [1](#fn:c)，这个数可能因为输入时删除掉前面的词而减少，或者如果用户手动 `shift+delete` 删除掉候选词也会 reset 成 0
*   `d` 权重，结合时间，综合计算一个权重，随着时间推移，d 权重会衰减
*   `t` 时间，记录该候选词最近一次的时间

Rime 有一些默认设置，比如上下键选词，但是很少人打字的时候会把手移动到上下键去选词。

这里记录一下我自己的一些适配，用以调整我的习惯。

在修改 Rime 配置是，可以打开 Rime 的日志，对于我使用 fcitx-rime ，那么对应的日志在

一般有 ERROR，WARNING, INFO 三个文件。每一次部署时最好同时开着文件观察错误情况。

在 `~/.config/fcitx/rime` 配置目录下，`vi default.custom.yaml`

```
patch:
  schema_list:
    - schema: luna_pinyin
    - schema: luna_pinyin_simp
    - schema: luna_pinyin_fluency
  menu:
    page_size: 6 
```

自定义其中的 `page_size` 即可。

Rime 自带繁简切换，菜单中 (Ctrl+grave) 就可以设置。

Rime 自带

将 [https://gist.github.com/2320943](https://gist.github.com/2320943) 作为模板保存到 `luna_pinyin_simp.custom.yaml` 、 `luna_pinyin_tw.custom.yaml` 或 `luna_pinyin_fluency.custom.yaml` 即可。

对于模糊音设置，官网有介绍 luna\_pinyin 的[实现方式](https://github.com/rime/home/wiki/CustomizationGuide#%E6%A8%A1%E7%B3%8A%E9%9F%B3)

对于我，这条加上还是很有必要的

*   `in` 和 `ing`

参考[链接](https://code.google.com/p/rimeime/wiki/CustomizationGuide#%E6%A8%A1%E7%B3%8A%E9%9F%B3)

配置英文字典，自定义名叫 english 的 translator，然后把这个 translator 添加到数据方案中。

```
# 加載 easy_en 依賴
"schema/dependencies/@next": easy_en
# 載入翻譯英文的碼表翻譯器，取名爲 english
"engine/translators/@next": table_translator@english
# english 翻譯器的設定項
english:
  dictionary: easy_en
  spelling_hints: 9
  enable_completion: true
  enable_sentence: false
  initial_quality: -3 
```

注意这个时候需要 Rime 的配置目录中有 `easy_en.dict.yaml` 这个码表文件。

安装完 Rime 后，会安装 `/usr/bin/rime_dict_manager` 工具和 `/usr/bin/rime_deployer` 工具。

在运行这两个工具前需要关闭正在使用的 Rime 输入法，否则会占用需要的文件，而出现错误：

```
E0114 17:38:47.016017  9869 level_db.cc:291] Error opening db 'luna_pinyin.userdb' read-only. 
```

管理工具，在 fcitx 的配置目录 `~/.config/fcitx/rime/` 下运行

会列出当前输入法配置方案。

导出词典

```
rime_dict_manager -e luna_pinyin export.txt 
```

完整参数：

```
➜ /usr/bin/rime_dict_manager
options:
		-l|--list
		-s|--sync
		-b|--backup dict_name
		-r|--restore xxx.userdb.txt
		-e|--export dict_name export.txt
		-i|--import dict_name import.txt 
```

看名字就知道 `rime_deployer` 是用来管理 Rime 部署相关操作的。

完整参数：

```
➜ /usr/bin/rime_deployer
options:
		--build [dest_dir [shared_data_dir]]
		--add-schema schema_id [...]
		--set-active-schema schema_id
		--compile x.schema.yaml [dest_dir [shared_data_dir]] 
```

自动生成主题皮肤

*   [https://rime.netlify.com](https://rime.netlify.com/)

material 质感的主题，很好看

*   [https://github.com/hrko99/fcitx-skin-material](https://github.com/hrko99/fcitx-skin-material)

fcitx 官方制作

*   [https://github.com/fcitx/fcitx-artwork](https://github.com/fcitx/fcitx-artwork)

Linux 终端配置 Rime 工具

*   [https://github.com/rime/plum](https://github.com/rime/plum)

OS X:

*   [https://github.com/neolee/SCU](https://github.com/neolee/SCU)

到这里，就会发现 Rime 已经能够满足日常的需求，我用全拼，培养一段时间之后词库也很满足我的需求了。而到搜狗官网去看其介绍，细胞词库，云端输入，自动纠错，多彩皮肤，长词联想，网址输入，大部分的功能 Rime 都能做到，并且没有隐私问题，数据完全掌握在自己手里何乐而不为。

之前一直使用 Google 拼音输入法，Google 拼音输入法能够导出一套用户长期积累的词库。我利用“[深蓝词库转换](https://code.google.com/p/imewlconverter/)“工具将 Google 拼音输入法导出的词库，大概 7 万多条转成 Rime 词库格式。然后开始菜单调出，小狼毫用户词典管理，选中 `luna_pinyin`，点击“导入文本码表”导入词库。瞬间就可以从 Google 拼音输入法转移到 Rime 输入法。导入文本码表只是针对 Google 拼音输入法中由用户积累的词汇。一般只建议将最为关键，带有词频信息的词库使用“导入文本码表”的方法导入到 Rime 输入法。

关于词库，Rime 输入法的词库有两部分组成。以下摘自贴吧：

*   一部分是由系统文本词库（一般以 xxx.dict.yaml 结尾）通过「重新部署 /deploy」生成的固态词典（一般以 xxx.table.bin 结尾），这部份词库因为在输入过程是固定不変的，所以存在用大量的词彚，也不允许用戸来直接删除。
*   另一部分就是记录我们用戸输入习惯的用戸词典（一般以 xxx.userdb.kct）结尾。这部份词库的词彚，正常情况下是由用戸输入的时候随时生成的；其词彚可以动态调整，数量理论上来说不会特别多，也允许用戸自行删除（shift+delete）。

佛振在设计用户词典时，没有考虑到有导入大词库的需求，就按照估计用戸可能积累起来的词彚量，把容量设置为「**十万**」规模以提升存储效率，超过这个量，性能则会下降。

佛振设计「【小狼毫】用戸词典管理」的初衷和真正目的，在於譲大家将自己従其他输入法中积累出来的用戸词彚，可以顺利地迁移到 rime 中。而不是譲你把其他输入法整个系统词库都搬进来。如今，「【小狼毫】用戸词典管理」这个功能和界面，已经被众多的小白同学稀里糊涂地滥用了。

如何正确的导入词库？

答：新增固态词典引用多份码表文件

过去一直没有简易的批量添加词汇做法，现在可以这样做：以【朙月拼音】为例，在输入方案裏指定一个新的词典名为

`luna_pinyin.extended.dict.yaml`

```
#luna_pinyin.custom.yaml
patch:
translator/dictionary: luna_pinyin.extended 
```

然后在用户目录创建一个词典文件 `luna_pinyin.extended.dict.yaml`

```
#Rime dictionary
---
name: luna_pinyin.extended
version: "2013.04.22"
sort: by_weight
use_preset_vocabulary: true
import_tables:
  - luna_pinyin
...
# table begins
鸹鸹！ gua gua 100 
```

这样一来，部署【朙月拼音】这个输入方案时，将编译 `luna_pinyin.extended` 这部词典，而这部词典除了导入【八股文】词汇表之外，还导入了名为 `luna_pinyin` 的词典文件，即 `luna_pinyin.dict.yaml` 。被导入的词典文件只取其码表，忽略 YAML 段。

被导入的码表与本词典自带的码表共同决定了编码集合。当然也可以：本文件的码表完全为空，只用来按需合并多个外部码表文件。

`luna_pinyin.extended` 这个词典的神奇之处是：虽然`luna_pinyin.schema.yaml` 已设置为加载名为 `luna_pinyin.extended` 的词典，但配套的用户词典名却是「`luna_pinyin`」，即 Rime 自动取句点之前的部分为用户词典名，以保证按以上方法增补了词汇之后，不至於因为改变词典名而抛弃原有的用户词典。

请注意，此法的设计用途是合并编码方案相同的多份词典文件，而不是用来将不同的编码混在一起。

具体的示例代码可参考 [https://gist.github.com/lotem/5443073](https://gist.github.com/lotem/5443073)

其中心思想提炼出来就是：

1.  先让输入方案引用一个新的系统词库码表（佛振同学在 `gist.github.com` 上的示例中是 `luna_pinyin.kunki.dict.yaml`），即给输入方案`luna_pinyin`（明月拼音）打一个补靪，将调用的词库重置为`luna_pinyin.kunki.dict.yaml`。
2.  创建一个`luna_pinyin.kunki.dict.yaml` 的文件，加入好你需要导入的词彚（如「瑾昀」等等）。并导入内置的系统词库（`import_tables: luna_pinyin`）。

其实佛振`import_tables`的这个做法，頪似於 C 语言编程中的 `#include` 头文件。其目的和工作机制都是一様的。目的是引用头文件（或是系统预设词库）竝添加上自己的内容；工作机制是在编译（或是重新部署的时候），将链接到的不同的文本文件合并成一个文件，并処理成二进制文件。

我另外要在佛振同学的基础上补充几点

1.  `luna_pinyin.custom.yaml` 和 `luna_pinyin.extended.dict.yaml`都要放入用戸文件夹中
2.  通过`import_tables` 的方法，不仅仅可以导入预设的词典，甚至可以导入其他的自定义词典

以笔者为例子，我在朙月拼音输入方案中设定的词库名叫 `luna_pinyin.extended.dict.yaml`。

而我 `luna_pinyin.extended.dict.yaml` 在文件头部分，除了系统预设词库之外，还导入了其他的细胞词库

```
import_tables:
  - luna_pinyin
  - luna_pinyin.extra_hanzi
  - luna_pinyin.sgmain
  - luna_pinyin.chat
  - luna_pinyin.net
  - luna_pinyin.user
  - luna_pinyin.cn_en
  - luna_pinyin.website
  - luna_pinyin.computer
  - luna_pinyin.place
  - luna_pinyin.shopping
  - luna_pinyin.sougou
  - luna_pinyin.kaomoji
  - mywords 
```

1.  码表中的词彚格式
    
    3.1 码表文件必须是 `utf-8` 无 bom 的编码。不能用 ansi，否则出来的词彚会乱码 3.2 Rime 对词彚的格式有着厳格的限定，其标凖形式是「`词彚<tab>ci hui<tab>100`」（方引号内部的部分，`<tab>`表示制表符（顕示为空白字符，不是空格））。
    

拼音码表的词彚格式是一个词彚占一行，不同的属性之间以制表符为间隔，编码之间以半角空格为间隔。従左往右依次是词彚、编码、词频。其中编码和词频是可省略的。也就是说

「`词彚<tab>ci hui`」、「`词彚<tab><tab>100`」、「`词彚`」

都是合法词库文件格式。

如果词频省略，那麼输入法会优先调用「八股文」（一个预设的中文语言模型，包含词彚和词频两穜属性）的词频，如果八股文找不到该词彚的词频，那麼这个词彚的词频就当成 0 来処理。

如果编码省略，那麼输入法在重新部署，将文本码表生成固态词典的时候，会根据词库中的单字来给词彚自动编码（如果是拼音的话，叫「给词彚注音」更妥帖） 比如词库中有

```
我<tab>wo
和<tab>he
你<tab>ni
我和你 
```

四个 item，那麼「我和你」这个省略了编码的词彚在生成固态词典的时候会自动被注音上「wo he ni」。其中有一个特别需要注意的地方，那就是処理多音字。对於含多音字的词彚，我们要侭量避免譲输入法给他自动注音，因为会帯来错误的读音（比如「重庆」读成「zhong qing」）所以一般含多音字的词彚都要最好标注上读音。如果实在没辧法弄到读音也没関系。因为 Rime 已经给多音字的罕见音做了降频処理。従而使得多音字的罕见音不会参与词彚的自动注音。

関於自动注音的具躰的细节可以看 rime 的 wiki，这裏我就不多说了。総而言之，我廃话那麼多，是为了譲大家了解 rime 词库的工作机制，其実就为了告诉大家两句话：「在导入词彚的时候，一般来说只要加纯汉字就够了。含多音字的词，系统词库一般都有，如果没有才要考虑给这个词注上音。」

另外，系统词库中，已经包含了完整的单字注音和罕用读音降频処理，大家可以放心地导入纯汉字词彚，不用太过担心。（所以一定给要记得`import_tables: luna_pinyin`，来使自定义码表获得系统词库中的单字注音、含多音字词彚注音以及系统词彚词频）

关於楼主配置的多个词库挂接的方法实例，可参考由 rime-aca 友情提供的「朙月拼音·扩充词库」

下载地址：https://bintray.com/rime-aca/dictionaries/luna\_pinyin.dict

参考：

*   [https://code.google.com/p/rime-aca/](https://code.google.com/p/rime-aca/)
*   [关于导入词库和深蓝词库转换](http://github.com/studyzy/imewlconverter/)

其他词库下载 [搜狗词库](http://cl.ly/033g2x3k2J05) [来源](http://blog.yesmryang.net/rime-setting/)

```
# weasel.custom.yaml
patch:
  "style/font_face": "华文行楷"  # 字體名稱，從記事本等處的系統字體對話框裏能看到
  "style/font_point": 16     # 字號，只認數字的，不認「五號」、「小五」這樣的

  style/horizontal: true      # 候選橫排
  style/inline_preedit: true  # 內嵌編碼（僅支持 TSF）
  style/display_tray_icon: true  # 顯示托盤圖標 
```

Rime 最让我惊讶的是还支持一些常见的快捷键操作，通过配合这些快捷键可以在输入很长一段句子的时候提升体验。

*   `ctrl+grave` (grave) tab 键上面，1 左边的那个键用来切换 Rime 输入方案
*   `shift+delete` 删除选中的候选词，一般用来调整不希望在候选词前的词
*   `ctrl+ n/p` 上下翻页选择候选词
*   `Ctrl+b/f` 类似于左箭头，右箭头，可以快速调整输入，在输入很长一段后调整之前的输入时非常有效
*   `Ctrl+a/e` 贯标快速跳转到句首或者句末
*   `Ctrl+d` 删除光标后内容
*   `Ctrl+h` 回退，删除光标前内容
*   `Ctrl+g` 清空输入
*   `Ctrl+k` 删词，等效于 Shift + delete（macOS 上可以使用 ⌘+k）
*   `-/+` 或者 `tab` 来翻页

更多的快捷键可以在 `default.yaml` 配置中看到。

遇到 Rime 在 Deploy 字典时

的问题，这些字典中的字符可能存在问题。

*   [https://github.com/rime/home/wiki/UserGuide](https://github.com/rime/home/wiki/UserGuide)
*   [https://www.byvoid.com/zht/blog/recommend-rime](https://www.byvoid.com/zht/blog/recommend-rime)
*   [https://mogeko.me/2018/031/](https://mogeko.me/2018/031/)
*   [https://jdhao.github.io/2019/02/18/rime\_configuration\_intro/](https://jdhao.github.io/2019/02/18/rime_configuration_intro/)
*   [https://laubonghaudoi.github.io/dialects/](https://laubonghaudoi.github.io/dialects/)
*   [https://mritd.me/2019/03/23/oh-my-rime/](https://mritd.me/2019/03/23/oh-my-rime/)
*   [https://withdewhua.space/2019/01/30/rime-configuration](https://withdewhua.space/2019/01/30/rime-configuration)
*   [https://www.dreamxu.com/install-config-squirrel/](https://www.dreamxu.com/install-config-squirrel/)
*   [https://kelvin.mbioq.com/guide-for-configuration-of-rime-input-method-on-linux.html](https://kelvin.mbioq.com/guide-for-configuration-of-rime-input-method-on-linux.html)
*   [https://github.com/vgist/rime-files](https://github.com/vgist/rime-files)
*   [https://scomper.me/gtd/shu-xu-guan-ci-ku-de-tong-bu-he-bei-fen-@vgow](https://scomper.me/gtd/shu-xu-guan-ci-ku-de-tong-bu-he-bei-fen-@vgow)
*   [图灵社区对 Rime 输入法作者佛振的采访](https://www.ituring.com.cn/article/118072)