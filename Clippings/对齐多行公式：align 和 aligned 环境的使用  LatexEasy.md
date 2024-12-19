---
title: "对齐多行公式：align 和 aligned 环境的使用 | LatexEasy"
source: "https://latexeasy.com/zh/document/14"
author:
published:
created: 2024-12-12
description: "对齐多行公式：align 和 aligned 环境的使用"
tags:
  - "clippings"
---

在排版数学公式时，有时需要对齐多行公式，以便更好地组织和展示数学表达式。LaTeX 提供了 `align` 和 `aligned` 环境，使你能够轻松地实现多行公式的对齐。本篇教程将向你介绍如何使用这两个环境来对齐多行公式。

### 使用 `align` 环境

`align` 环境可以用于对齐多行公式，并在公式的对齐点添加对齐标记（通常是等号）。每行公式都以 `&` 符号标记对齐点，而换行使用 `\\`。

例如，要对齐以下两个等式：

`a + b &= c \\ d &= e + f`

你可以这样使用 `align` 环境：

```latex 
\begin{align} 
a + b &= c \\ 
d &= e + f 
\end{align}
```

效果如下:

$$
\begin{align} 
a + b &= c \\ 
d &= e + f 
\end{align}
$$

### 使用 `aligned` 环境

`aligned` 环境允许在数学环境内嵌套使用，用于对齐多行公式，但不添加额外的对齐标记。通常，你可以将 `aligned` 环境嵌套在其他数学环境中，如 `equation`。

例如，要在一个编号的等式中对齐多行公式：

```latex
\begin{equation} 
\begin{aligned} 
a + b &= c \\ 
d &= e + f 
\end{aligned} 
\end{equation}
```

效果如下:

$$
\begin{equation} 
\begin{aligned} 
a + b &= c \\ 
d &= e + f 
\end{aligned} 
\end{equation}
$$

### 对齐点的控制

在 `align` 或 `aligned` 环境中，你可以使用 `&` 符号来标记对齐点，以及 `\\` 来换行。通过调整 `&` 符号的位置，你可以控制对齐点的位置。

例如，要在等号对齐点对齐：

```latex
\begin{align} 
a &= b + c \\ 
d &= e + f 
\end{align}
```

效果如下:

$$
\begin{align} 
a &= b + c \\ 
d &= e + f 
\end{align}
$$

### 总结

通过使用 `align` 和 `aligned` 环境，你可以在 LaTeX 中轻松地对齐多行公式，从而更好地组织和呈现数学表达式。这两个环境分别适用于需要和不需要对齐标记的情况。在下一篇教程中，我们将继续探讨其他数学排版技巧和环境。