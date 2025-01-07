---
title: "git使用指定的私钥private key"
source: "https://blog.blessedbin.com/2024/02/26/git%E4%BD%BF%E7%94%A8%E6%8C%87%E5%AE%9A%E7%9A%84%E7%A7%81%E9%92%A5private-key/"
author:
  - "[[捌号视角]]"
published: 2024-02-26
created: 2024-12-31
description: "Using a Specific SSH Private Key When Using Git Command一、背景当不同的git库需要使用不同的private key的时候，可在运行git命令的时候指定私钥 private key。 二、两种方式使用SSH配置文件我们可以通过SSH配置文件来指定在git clone过程中使用特定的私钥。 具体来说，我们可以在~/.ssh/config文件中为不"
tags:
  - "clippings"
---
## Using a Specific SSH Private Key When Using Git Command
## 一、背景

当不同的git库需要使用不同的private key的时候，可在运行git命令的时候指定私钥 private key。

### 二、两种方式

#### **使用SSH配置文件**

我们可以通过SSH配置文件来指定在git clone过程中使用特定的私钥。

**具体来说，我们可以在~/.ssh/config文件中为不同的私钥创建两个单独的主机。然后，在git clone期间，根据我们想要使用的密钥，我们可以在SSH连接字符串中指定不同的主机。**

如：

| ``` 12345678 ``` | ``` cat ~/.ssh/configHost github-work    HostName github.com    IdentityFile ~/.ssh/id_rsa_workHost github-personal    HostName github.com    IdentityFile ~/.ssh/id_rsa_personal ``` |
| --- | --- |

现在，可以通过id\_rsa\_work私钥访问的存储库上运行git clone命令，我们可以使用github-work作为其主机来指定SSH连接字符串。

如：

| ``` 12 ``` | ``` git clone git@github-work:corporateA/webapp.gitgit clone git@github-personal:bob/blog.git ``` |
| --- | --- |

#### 使用core.sshCommand

Git仓库提供了一个可配置的选项core.sshCommand。当运行任何需要SSH隧道的命令时，此配置将覆盖默认的SSH命令。

**配置 core.sshCommand**

| ``` 1 ``` | ``` git clone -c "core.sshCommand=ssh -i ~/.ssh/id_rsa_work" git@github.com:corporateA/webapp.git ``` |
| --- | --- |

以上例子中，使用-c选项在运行时覆盖core.sshCommand。具体来说，我们使用-i选项将SSH命令更改为指向我们想要使用的私钥。

**在存储库级别上持久化core.sshCommand**

我们可以使用git config命令将配置覆盖持久化到存储库级别，而不是每次重复配置覆盖

| ``` 1 ``` | ``` git config core.sshCommand "ssh -i ~/.ssh/id_rsa_work" ``` |
| --- | --- |

该命令将core.sshCommand配置持久化到仓库中。这意味着Git将在后续的调用中使用该SSH命令。

可以使用以下命令验证是否配置成功

| ``` 12 ``` | ``` $ git config --get core.sshCommandssh -i ~/.ssh/id_rsa_work ``` |
| --- | --- |