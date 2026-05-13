#tmux 

```bash
eval $(tmux showenv -s | grep -E '^(SSH|DISPLAY)')
```

在 tmux 配置文件中添加：

```
# 让 tmux 运行 update-environment 命令来抓取这些变量
set -g update-environment "DISPLAY SSH_ASKPASS SSH_AUTH_SOCK SSH_AGENT_PID SSH_CONNECTION WINDOWID XAUTHORITY"
```

然后刷新 tmux 配置使之生效，然后执行上一句

# 参考

[修复损坏的 SSH / X11 转发与 tmux（以及 fish！）| 作者：Craig Younkins | Medium --- Fixing Broken SSH / X11 Forwarding with tmux (and fish!) | by Craig Younkins | Medium](https://cyounkins.medium.com/fixing-broken-ssh-x11-forwarding-with-tmux-and-fish-32500642f6f2)