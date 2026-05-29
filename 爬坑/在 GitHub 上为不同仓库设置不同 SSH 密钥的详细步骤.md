# 《在 GitHub 上为不同仓库设置不同 SSH 密钥的详细步骤》

在 GitHub 上为不同仓库设置不同 SSH 密钥，可以通过配置 SSH 配置文件（`~/.ssh/config`）来实现。这种方法允许你根据不同的仓库使用不同的密钥对，提高安全性和管理效率。以下是详细步骤：

### **1. 生成新的 SSH 密钥对**

如果你还没有多个 SSH 密钥，可以为每个需要区分的仓库生成新的密钥对。例如：

```bash
# 为工作仓库生成密钥
ssh-keygen -t ed25519 -C "your_work_email@example.com" -f ~/.ssh/id_ed25519_work


# 为个人仓库生成密钥
ssh-keygen -t ed25519 -C "your_personal_email@example.com" -f ~/.ssh/id_ed25519_personal
```

按提示输入密码（可选），生成后会得到两个密钥文件：

*   `id_ed25519_work`（私钥）和 `id_ed25519_work.pub`（公钥）
*   `id_ed25519_personal`（私钥）和 `id_ed25519_personal.pub`（公钥）


### **2. 将公钥添加到 GitHub**

1.  复制公钥内容（例如工作密钥）：

```bash
cat ~/.ssh/id_ed25519_work.pub
```

1.  登录 GitHub，进入 **Settings → SSH and GPG keys**。
2.  点击 **New SSH key**，粘贴公钥内容，并为其命名（如 `Work Laptop - Work Repo`）。
3.  重复上述步骤，添加个人仓库的公钥。

### **3. 配置 SSH 以使用不同密钥**

编辑 SSH 配置文件 `~/.ssh/config`（如果不存在则创建）：

```bash
nano ~/.ssh/config
```

添加以下内容（根据实际情况修改）：

```yaml
# GitHub Work Account
Host github.com-work
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_work
    IdentitiesOnly yes
# GitHub Personal Accounte
Host github.com-personal
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_personal
    IdentitiesOnly yes
```

**参数说明**：
*   `Host`：自定义的主机别名（用于 Git 命令）。
*   `HostName`：实际连接的服务器地址（保持 `gith[ubuntu24.04新机装机](../DevelopTools/ubuntu24.04新机装机.md)ub.com`）。
*   `IdentityFile`：指定使用的私钥路径。
*   `IdentitiesOnly yes`：强制使用指定的密钥，避免冲突。


### **4. 使用别名克隆仓库**

克隆仓库时，将默认的 `github.com` 替换为你在 `config` 中定义的 `Host` 别名：

```bash
# 克隆工作仓库（使用工作密钥）
git clone git@github.com-work:your_username/work-repo.git
# 克隆个人仓库（使用个人密钥）
git clone git@github.com-personal:your_username/personal-repo.git
```

### **5. 验证配置**

测试 SSH 连接是否正常：

```bash
# 测试工作密钥
ssh -T git@github.com-work
# 测试个人密钥
ssh -T git@github.com-personal
```

如果看到类似 `Hi your_username! You've successfully authenticated…` 的提示，则配置成功。

### **6. 已有仓库的处理**

如果需要为已克隆的仓库更改 SSH 密钥，修改其远程地址：

```bash
# 进入仓库目录
cd /path/to/your/repo
# 修改远程地址（以工作仓库为例）
git remote set-url origin git@github.com-work:your_username/work-repo.git
```

### **注意事项**
1.  **密钥安全**：私钥不要泄露，建议设置密码保护。
2.  **配置文件权限**：确保 `~/.ssh/config` 的权限为 `600`（`chmod 600 ~/.ssh/config`）。
3.  **密钥缓存**：如果使用 macOS，可通过 `ssh-add -K` 将密钥添加到钥匙串。
4.  **多账户限制**：同一邮箱不能用于多个 GitHub 账户的 SSH 密钥。

通过这种方式，你可以灵活管理不同仓库的访问权限，提高安全性。
Hq