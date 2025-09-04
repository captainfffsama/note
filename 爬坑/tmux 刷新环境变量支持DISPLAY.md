#tmux 

```bash
eval $(tmux showenv -s | grep -E '^(SSH|DISPLAY)')`
```

# 参考

[修复损坏的 SSH / X11 转发与 tmux（以及 fish！）| 作者：Craig Younkins | Medium --- Fixing Broken SSH / X11 Forwarding with tmux (and fish!) | by Craig Younkins | Medium](https://cyounkins.medium.com/fixing-broken-ssh-x11-forwarding-with-tmux-and-fish-32500642f6f2)