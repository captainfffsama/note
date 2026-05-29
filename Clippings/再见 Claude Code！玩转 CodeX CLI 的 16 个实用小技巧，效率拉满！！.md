·
---
title: "再见 Claude Code！玩转 CodeX CLI 的 16 个实用小技巧，效率拉满！！"
source: "https://www.cnblogs.com/javastack/p/19113665"
author:
  - "[[Java技术栈]]"
published: 2025-09-26
created: 2026-03-10
description: "大家好，我是R哥。 最近用上了 CodeX CLI，替代了 Claude Code，原因不多说，看这篇： 再见 Claude Code，我选择了 Codex！真香！！ 今天，我再来分享一波我实战中积累的 CodeX CLI 实用小技巧，不管你是新手刚入坑，还是老司机，这些技巧都能帮你用得更溜、更爽、"
tags:
  - "clippings"
---
大家好，我是R哥。

最近用上了 CodeX CLI，替代了 Claude Code，原因不多说，看这篇：

> [再见 Claude Code，我选择了 Codex！真香！！](https://www.javastack.cn/bye-claude-code-choose-codex/)

今天，我再来分享一波我实战中积累的 **CodeX CLI 实用小技巧** ，不管你是新手刚入坑，还是老司机，这些技巧都能帮你用得更溜、更爽、更高效。

这些都是我自己真正在项目中踩过坑、总结出的实战经验，不管你平时在公司写代码，还是想让 CodeX 整点副业的活，这波技巧都值得你收藏。

那我们开干吧，没空看，关注收藏一波！

## 1、使用别名

学会使用别名，快速启动工作。

如下所示：

> alias codex='codex -m gpt-5-codex -c model\_reasoning\_effort="high" -c model\_reasoning\_summary\_format=experimental --search --dangerously-bypass-approvals-and-sandbox'

这样只是临时在当前会话生效，关掉就没有了，如果要永久生效，建议在本机环境中配置。

Mac 参考配置如下：

> echo "alias codex='codex -m gpt-5-codex -c model\_reasoning\_effort="high" -c model\_reasoning\_summary\_format=experimental --search --dangerously-bypass-approvals-and-sandbox'" >> ~/.zshrc  
> source ~/.zshrc

这样在启动的时候就配置好模型等参数了，而不需要启动之后再调整，相关参数在下面会有详细说明。

## 2、使用快捷命令

在 CodeX 中使用 `/` 就能调出所有已支持的快捷命令：

![](https://www.javastack.cn/images/img7/20250910120702303.png)

命令功能解释如下表所示：

| 命令 | 中文说明 |
| --- | --- |
| /model | 切换模型和推理等级 |
| /approvals | 设置授权模式 |
| /new | 开启新的会话 |
| /init | 初始化 **AGENTS.md** 指导文件 |
| /compact | 上下文压缩，避免触发上下文限制 |
| /diff | 显示 git 差异 |
| /mention | 引用某个文件 |
| /status | 显示当前会话配置和 Token 用量 |

## 3、快速换行

使用 `Option` + `Enter` 或者 `Control` + `J` 可以快速换行：

![](https://www.javastack.cn/images/img7/20250910163143600.png)

有时候输入的提示词太长，想换行输入，看起来更清晰，这招特别有用。

## 4、中断请求/退出会话

在 CodeX 正在工作时，你随时可以按 `ESC` 键或者 `Control` + `C` 键中断当前的请求：

![](https://www.javastack.cn/images/img7/20250910163913664.png)

再按一次 `Control` + `C` 键，或者输入 `/quit` 可以退出当前会话。

## 5、通过 API 使用

如果你没有订阅付费计划， **免费账户也可以通过付费 API 的形式使用 CodeX** ，你可以通过修改 codex 配置文件中的 `preferred_auth_method` 配置来更换为 API 使用认证方式。

### 费用说明

以下是 **GPT-5** 的 API 价格：

![](https://www.javastack.cn/images/img7/20250911093613547.png)

以下是其他 GPT 的 API 价格：

![](https://www.javastack.cn/images/img7/20250911093711316.png)

更多费用明细查看官方说明：

> [https://openai.com/api/pricing/](https://openai.com/api/pricing/)

### 切换 API 认证

修改配置文件：

> ~/.codex/config.toml

添加以下 API 认证配置：

```
preferred_auth_method = "apikey"
```

你也可以通过 CLI 命令行工具临时覆盖这个设置：

> codex --config preferred\_auth\_method="apikey"

你可以通过运行以下命令返回到 ChatGPT 默认认证方式：

> codex --config preferred\_auth\_method="chatgpt"

你可以根据需要来回切换，比如说，如果你的 ChatGPT 账号额度用完了，你就可以切换 API 方式来使用 CodeX。

## 6、初始化项目指导文件

CodeX 中的 `AGENTS.md` 是一个简单又开放的格式，专门用来指导 CodeX 更好的干活。

可以把 `AGENTS.md` 想象成是给 Agents 准备的 README，它提供了一个专门的、可预测的地方，用来提供上下文和指令，帮助 AI 编码 Agents 更好地完成你的项目。

> 详细介绍： [https://agents.md/](https://agents.md/)

![](https://www.javastack.cn/images/img7/20250910160150454.png)

通过 `/init` 命令，初始化一个项目的 **AGENTS.md** 指导文件：

![](https://www.javastack.cn/images/img7/20250910155930875.png)

默认使用的文件是英文的，可以手动把它转换成中文：

> 把AGENTS.md转换成中文

![](https://www.javastack.cn/images/img7/20250910161126162.png)

## 7、切换模型与推理等级

Codex 默认搭配的是 OpenAI 最牛的代码专用模型 **gpt-5-codex** ，默认的推理等级是： **中等** ，gpt-5-codex 是 CodeX 专用的代码模型，比 GPT-5 更强。

如图所示：

![](https://www.javastack.cn/images/img7/20250917151531632.png)

可以用 `/model` 命令切换模式及推理等级：

![](https://www.javastack.cn/images/img7/20250916162706763.png)

也可以通过在启动 codex 的时候，加上 `-m` 或者 `--model` 参数来切换指定模型。

比如切换到 `gpt-5` 模型：

> codex --model gpt-5 -c model\_reasoning\_effort="high"

也可以是这样：

> codex -m gpt-5 -c model\_reasoning\_effort="high"

这样，如果使用 `/model` 命令无法切换到旧模型（比如：o4-mini），就可以使用这招来切换了。

另外，还支持 `-c model_reasoning_summary_format` 参数，强制推理总结格式，支持 `none` | `experimental` 两个值，即默认格式/实验性格式。

## 8、切换授权模式

Codex 的默认授权模式为 **auto** ，Codex 可以自动读取文件、进行编辑并在工作目录中运行命令。不过， **处理工作目录外的文件或访问网络，它会需要你的同意** 。

如果你只想随便聊聊，或者在真正开始之前先规划一下？

用 `/approvals` 命令切换到 `Read Only` 只读模式就行啦：

![](https://www.javastack.cn/images/img7/20250910114442565.png)

如果你需要 Codex 在未经你允许的情况下，就能 **读取文件、编辑内容、运行命令，甚至处理工作目录外的文件或者访问网络** ，那你可以用 `Full Access` 完全访问模式。

不过用之前可得小心点！

来对比下三种模式的差异：

| 权限项 | Auto（默认） | Read Only | Full Access |
| --- | --- | --- | --- |
| 读取文件 | ✅ | ✅ | ✅ |
| 编辑文件 | ✅ | ❌ | ✅ |
| 在工作目录运行命令 | ✅ | ❌ | ✅ |
| 访问工作目录外文件 | ❌（需确认） | ❌ | ✅ |
| 访问网络 | ❌（需确认） | ❌ | ✅ |

根据自己的开发环境及项目，再选择合适的授权模式吧。

另外，还支持通过 `Flags` 参数来精细控制权限：

| 模式 | 标志 | 说明 |
| --- | --- | --- |
| 自动（默认） | 无需标志，默认值 | Codex 可以读取文件、编辑文件并在工作区运行命令。Codex 在运行沙箱外的命令时会请求批准。 |
| 只读 | `--sandbox read-only --ask-for-approval never` | Codex 只能读取文件；从不请求批准。 |
| 自动编辑，但运行不可信命令时需批准 | `--sandbox workspace-write --ask-for-approval untrusted` | Codex 可以读取和编辑文件，但在运行不可信命令之前会请求批准。 |
| 危险的完全访问 | `--dangerously-bypass-approvals-and-sandbox` （别名： `--yolo` ） | 无沙箱、无批准（不推荐）。 |

这样，在启动 CodeX 的时候通过指定 Flags 参数就能使用指定的授权模式了。

## 9、网络搜索

有了 Web search 网络搜索，模型就能访问互联网上的最新信息，让模型在生成回复之前，先上网搜搜最新的信息，并提供带有出处的答案。

想启用这个功能，传递 `--search` 参数即可：

> codex --search

这样可以仅使用网络搜索，而避免给 agent 完全不受限制的网络访问权限。

## 10、引用文件

你可以通过 `@` 这种方式快速引用项目中的任何文件：

![](https://www.javastack.cn/images/img7/20250910124233947.png)

在 IDE 插件使用更方便：

![](https://www.javastack.cn/images/img7/20250910123700399.png)

引用具体文件，可以让 CodeX 工作更高效，也可以防止它改错文件。

## 11、输入图像

你可以直接把图片粘贴到编辑器里，这样就能把它们添加到你的 prompt 中：

![](https://www.javastack.cn/images/img7/20250910115609986.png)

当然，你也可以通过 CLI，使用 `-i` 或者 `--image` 这个 flag 来附加图片文件：

> codex -i screenshot.png "解释一下这个代码"
> 
> codex --image img1.png,img2.jpg "总结一下这些图表"

## 12、脚本功能

你还可以用 `exec` 命令，以非交互方式运行 Codex：

> codex exec "修复这个报错的问题"

## 13、上下文压缩

执行 `/compact` 命令可压缩上下文，避免触发上下文限制。

如图所示，输入框下面会显示当前剩余上下文长度：

![](https://www.javastack.cn/images/img7/20250910162518990.png)

## 14、查看配置和 Token 用量

使用 `status` 命令可以查看当前会话的配置和 Token 用量：

![](https://www.javastack.cn/images/img7/20250910162930752.png)

如图所示，显示了我的 **账户信息、模型信息、Token 使用量信息** 等等。

## 15、MCP 集成

### 配置 MCP

Codex CLI 可以通过在 `~/.codex/config.toml` 中定义一个 `mcp_servers` 部分来配置 MCP，和 Claude 和 Cursor 在各自的 JSON 配置文件中定义 `mcpServers` 一样，但是 Codex 的格式略有不同，它使用 TOML，而不是 JSON。

比如我添加了以下几个 MCP：

```toml
[mcp_servers.context7]
command = "npx"
args = ["-y", "@upstash/context7-mcp"]
env = { "test" = "123456" }

[mcp_servers.puppeteer]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-puppeteer"]
env = { "test" = "123456" }
```

### 验证 MCP

目前来说，Codex 还没有提供专门的命令来验证 MCP 服务器的集成情况，不像 Claude Code / Gemini CLI 能提供详细的 MCP 连接信息，相信后续迭代会添加上。

不过，要是在启动 Codex 时，如果连不上你配置的 MCP server，就会给出错误信息。

比如我故意把 `@upstash/context7-mcp` 改成 `@upstash/context7-mcp1` 后，再执行 codex：

![](https://www.javastack.cn/images/img7/20250917161811370.png)

要是没看到这个报错，那就可以认为 MCP 集成没问题了。

### 使用 MCP

比如，我来测试下使用 `context7` 这个 MCP Server：

> 请使用context7写一段最新的java25的switch代码

![](https://www.javastack.cn/images/img7/20250917162614223.png)

![](https://www.javastack.cn/images/img7/20250917162658787.png)

如图所示，控制台已经显示成功调用 `context7` 工具，并成功输出了代码。

## 16、更多配置

Codex CLI 还支持超多配置选项，偏好设置都存在 `~/.codex/config.toml` 这个配置文件里，一些命令行设置的，或者通过 `/` 设置的个性化参数都可以在这里进行配置。

如果使用了 VS Code 中的 CODEX 插件，可以在设置菜单中打开并编辑这个文件：

![](https://www.javastack.cn/images/img7/20250917155101537.png)

示例配置如下：

```toml
model = "gpt-5-codex"
[projects."/Users/XX/project1"]
trust_level = "trusted"

[projects."/Users/XX/project2"]
trust_level = "trusted"

...
```

更多详细配置参考官方文档：

> [https://github.com/openai/codex/blob/main/docs/config.md](https://github.com/openai/codex/blob/main/docs/config.md)

---

好了，这次的分享就到了～

以上就是我在实际使用 CodeX 编程时的一些 **高效技巧和避坑心得** ，真的都是无保留实践总结。

不过，现在 CodeX CLI 还有一个不太方便的操作，那就是不能管理和恢复会话，相信在后续的版本中都会补齐，后续更新。

未完待续，接下来会继续分享更多 CodeX 的心得体验、高级使用技巧，公众号持续分享 AI 实战干货，关注「 **AI技术宅** 」公众号和我一起学 AI。

**AI 不会淘汰程序员，但不会用 AI 的除外，会用 AI 的程序员才有未来！**

> **版权声明：** 本文系公众号 "AI技术宅" 原创，转载、引用本文内容请注明出处，抄袭、洗稿一律投诉侵权，后果自负，并保留追究其法律责任的权利。

posted @ [Java技术栈](https://www.cnblogs.com/javastack) 阅读(21347)  评论(0) [收藏](https://www.cnblogs.com/javastack/p/) [举报](https://www.cnblogs.com/javastack/p/)