# 《流匹配与扩散模型导论》Markdown 中英文版

原文：**An Introduction to Flow Matching and Diffusion Models**  
作者：Peter Holderrieth、Ezra Erives  
版本：**arXiv:2506.02070v3**，2026 年 3 月 18 日

- [论文主页](https://arxiv.org/abs/2506.02070v3)
- [LaTeX 源码](https://arxiv.org/src/2506.02070v3)
- [课程网站](https://diffusion.csail.mit.edu/)
- 原文许可：[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

## 文件

| 文件或目录 | 内容 |
| --- | --- |
| [paper.en.md](paper.en.md) | 英文全文，含 7 节正文、参考文献、附录 A–E、图注及脚注 |
| [paper.zh.md](paper.zh.md) | 对应的完整中文译文 |
| images/ | 两版共用的 25 个 PNG 图片文件，均从原始源码提取或由源码中的 PDF 插图转换 |
| [preview.en.html](preview.en.html) | 英文离线阅读预览 |
| [preview.zh.html](preview.zh.html) | 中文离线阅读预览 |
| assets/ | 离线预览使用的 KaTeX 样式与字体，附其许可证 |
| [validation.json](validation.json) | 公式渲染、表格、图片路径与内部链接的检查结果 |

## 阅读与格式

请将 Markdown 文件与 `images/` 保持在同一个目录中，或直接解压整个 ZIP 后阅读。图片均采用 `images/文件名.png` 的相对路径；HTML 预览也可离线打开。

Markdown 需使用支持 LaTeX 数学公式的阅读器。行内公式使用单 `$`，独立公式使用双 `$$`；多行推导使用 `aligned` 等兼容环境。源码中的数学宏已展开，并调整了颜色语法、嵌套数学定界符等影响 Markdown 渲染的细节。

表格以 Markdown 管道表格保存。8 个算法也以“步骤—操作”表格呈现；算法中的 9 处独立长公式放在相应表格下方，通过链接与步骤对应，避免破坏表格布局。原文用于排列插图的 5 个表格也转换为 Markdown 表格。

中文版本翻译了正文、证明、图注、算法、脚注与公式中的自然语言说明。公式的数学结构与符号保持对应；文献条目、模型专名、常用缩写以及原图内的文字保留原文。文稿按指定 v3 版本转换，不另作学术勘误。

## 检查结果

- 全部 655 个待译文本单元均已翻译，逐段核对公式占位符与链接目标。
- 每版 1,594 处公式均通过 KaTeX 渲染与表达式完整性检查。
- 每版包含 13 个 Markdown 表格、25 个图片引用；所有图片均为有效的相对路径。
- 目录、正文交叉引用与算法公式链接均有有效目标；无未替换占位符。

工作目录中另外保留了下载的源码、转换脚本与翻译中间文件；便携 ZIP 仅包含上表列出的阅读文件和资源。
