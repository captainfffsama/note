#爬坑 

[toc]

# Dockered 代理

Dockerhub 被墙, 需要挂代理才能 pull docker.

## 做法 1

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo touch /etc/systemd/system/docker.service.d/proxy.conf
```

在 `proxy.conf` 中添加以下内容, 相应的 IP 和端口改成本机的带来, 如 clash 等.

```
[Service] 
Environment="HTTP_PROXY=http://127.0.0.1:7890/" Environment="HTTPS_PROXY=http://127.0.0.1:7890/" Environment="NO_PROXY=localhost,127.0.0.1,.example.com"
```

重启 docker

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

使用以下命令可以查看代理参数是否设置成功:

```bash
sudo systemctl show --property=Environment docker
```

## 做法 2

在 `docker build` 命令中追加如下参数:

```bash
docker build . \
--build-arg "HTTP_PROXY=http://proxy.example.com:8080/" \
--build-arg "HTTPS_PROXY=http://proxy.example.com:8080/" \
--build-arg "NO_PROXY=localhost,127.0.0.1,.example.com" \
-t your/image:tag
```

# 容器内部代理

在 `~/.docker/config.json` 中添加如下参数:

```bash
{
 "proxies":
 {
   "default":
   {
     "httpProxy": "http://proxy.example.com:8080",
     "httpsProxy": "http://proxy.example.com:8080",
     "noProxy": "localhost,127.0.0.1,.example.com"
   }
 }
}
```

# 代理
- <https://www.cnblogs.com/Chary/p/18096678>