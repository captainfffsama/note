一些 pytorch 的 wheel 是使用 `-D_GLIBCXX_USE_CXX11_ABI=0` 编译的, 常见于从 <pytorch.org> 上下载的. 而有些, 比如在 nvcr docker 镜像中编译的 pytorch, 编译时使用了 `-D_GLIBCXX_USE_CXX11_ABI=1`, 具体可以通过 `torch._C._GLIBCXX_USE_CXX11_ABI` 来查看本地 pytorch 的版本兼容.

# 参考
- <https://github.com/Dao-AILab/flash-attention/issues/457>
- 