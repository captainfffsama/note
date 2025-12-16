#Isaacsim 

[toc]

# 报错 errno=28 /No space left on device
## 现象

执行 `python show_scene.py` 时报错：

```text
2025-12-10T02:00:08Z [8,448ms] [Error] [carb] Failed to create change watch for `/home/hc-em/miniforge3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.physx.fabric-107.3.26+107.3.3.lx64.r.cp311.u353/config`: errno=28 /No space left on device
```

## 解决方法

这是一个非常经典的 Linux 系统错误，特别是在运行像 Isaac Sim (Omniverse) 这样需要加载大量文件和扩展的软件时。

报错信息 `errno=28 / No space left on device` 在这里具有误导性。结合前面的 `Failed to create change watch`，这通常**不是指你的硬盘空间满了**，而是指 **Linux 内核的文件监控（Inotify Watches）数量达到了上限**。

Isaac Sim 启动时会监控成千上万个扩展和配置文件，默认的 Linux 限制（通常是 8192）不够用。

请按照以下步骤解决：

### 核心解决方案：增加 Inotify Watcher 限制（最可能的修复方法）

#### 1. 临时修复（立即生效，重启失效）

在终端中执行以下命令，将限制提高到 524288（Omniverse 推荐值）：

```bash
sudo sysctl -w fs.inotify.max_user_watches=524288
```

执行完后，重新运行你的 IsaacLab 任务，应该就不会报错了。

#### 2. 永久修复（重启依然生效）

为了避免下次重启电脑后再次报错，建议将其写入系统配置。

1. 编辑 sysctl 配置文件：

    ```bash
    sudo nano /etc/sysctl.conf
    ```

2. 在文件末尾添加一行：

    ```Plaintext
    fs.inotify.max_user_watches = 524288
    ```

3. 保存并退出（在 nano 中按 `Ctrl+O` -> `Enter` -> `Ctrl+X`）。
4. 加载配置：

    ```bash
    sudo sysctl -p
    ```

### 备选排查：真的硬盘满了？

虽然概率较小，但如果上面的方法无效，可能真的是物理存储空间或 Inode 耗尽。

#### 1. 检查磁盘空间

运行以下命令查看你的根目录或 `/home` 目录是否已满：

```bash
df -h
```

如果 Use% 是 100%，你需要清理一些文件。Isaac Sim 的缓存通常很大，可以尝试清理：

- `~/.cache/ov`
- `~/.cache/nvidia/GLCache`
- `~/.nv/ComputeCache`
#### 2. 检查 Inode 数量

有时候空间有剩余，但小文件太多导致“文件索引（Inode）”耗尽：

```bash
df -i
```

如果 `/home` 对应的 IUse% 是 100%，你需要删除大量小文件（比如旧的 python 环境或编译碎片）。

### 总结
**99% 的情况下，执行“临时修复”中的 `sudo sysctl` 命令即可解决此问题。**


# 源码安装 Isaaclab 报错ERROR: Failed building wheel for egl_probe
## 现象
执行源码安装 isaaclab 时报错如下：
```text
Building wheels for collected packages: isaaclab_mimic, robomimic, egl_probe
  Building editable for isaaclab_mimic (pyproject.toml) ... done
  Created wheel for isaaclab_mimic: filename=isaaclab_mimic-1.0.15-0.editable-py3-none-any.whl size=3136 sha256=4e2f4b28e9ef564deabd544c3618a3f4287b84679d7b895cfe0890b2c75034f9
  Stored in directory: /tmp/pip-ephem-wheel-cache-ndtpmfnx/wheels/89/c1/6e/7373b8a08ca9c25d6a20337735482a5320f59a4380deb27011
  Building wheel for robomimic (pyproject.toml) ... done
  Created wheel for robomimic: filename=robomimic-0.4.0-py3-none-any.whl size=257616 sha256=e18c800ac28e2ca0cc6fc392dc4ca944305ae391bc7498cc294406c345e8f768
  Stored in directory: /tmp/pip-ephem-wheel-cache-ndtpmfnx/wheels/75/03/ee/bf8084201b458ed61e15d6f472d1d94d2e25b4d7a0be47e78c
  Building wheel for egl_probe (pyproject.toml) ... error
  error: subprocess-exited-with-error

  × Building wheel for egl_probe (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [374 lines of output]
      /tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/_distutils/dist.py:289: UserWarning: Unknown distribution option: 'tests_require'
        warnings.warn(msg)
      running bdist_wheel
      running build
      running build_py
      creating build/lib.linux-x86_64-cpython-311/egl_probe
      copying egl_probe/get_available_devices.py -> build/lib.linux-x86_64-cpython-311/egl_probe
      copying egl_probe/__init__.py -> build/lib.linux-x86_64-cpython-311/egl_probe
      running egg_info
      writing egl_probe.egg-info/PKG-INFO
      writing dependency_links to egl_probe.egg-info/dependency_links.txt
      writing top-level names to egl_probe.egg-info/top_level.txt
      reading manifest file 'egl_probe.egg-info/SOURCES.txt'
      reading manifest template 'MANIFEST.in'
      adding license file 'LICENSE'
      writing manifest file 'egl_probe.egg-info/SOURCES.txt'
      /tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/command/build_py.py:212: _Warning: Package 'egl_probe.glad' is absent from the `packages` configuration.
      !!

              ********************************************************************************
              ############################
              # Package would be ignored #
              ############################
              Python recognizes 'egl_probe.glad' as an importable package[^1],
              but it is absent from setuptools' `packages` configuration.

              This leads to an ambiguous overall configuration. If you want to distribute this
              package, please make sure that 'egl_probe.glad' is explicitly added
              to the `packages` configuration field.

              Alternatively, you can also rely on setuptools' discovery methods
              (for example by using `find_namespace_packages(...)`/`find_namespace:`
              instead of `find_packages(...)`/`find:`).

              You can read more about "package discovery" on setuptools documentation page:

              - https://setuptools.pypa.io/en/latest/userguide/package_discovery.html

              If you don't want 'egl_probe.glad' to be distributed and are
              already explicitly excluding 'egl_probe.glad' via
              `find_namespace_packages(...)/find_namespace` or `find_packages(...)/find`,
              you can try to use `exclude_package_data`, or `include-package-data=False` in
              combination with a more fine grained `package-data` configuration.

              You can read more about "package data files" on setuptools documentation page:

              - https://setuptools.pypa.io/en/latest/userguide/datafiles.html


              [^1]: For Python, any directory (with suitable naming) can be imported,
                    even if it does not contain any `.py` files.
                    On the other hand, currently there is no concept of package data
                    directory, all directories are treated like packages.
              

      !!
        check.warn(importable)
      /tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/command/build_py.py:212: _Warning: Package 'egl_probe.glad.X11' is absent from the `packages` configuration.

            !!
....
      running build_ext
      CMake Error at CMakeLists.txt:1 (cmake_minimum_required):
        Compatibility with CMake < 3.5 has been removed from CMake.

        Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
        to tell CMake that the project requires at least <min> but has been updated
        to work with policies introduced by <max> or earlier.

        Or, add -DCMAKE_POLICY_VERSION_MINIMUM=3.5 to try configuring anyway.


      -- Configuring incomplete, errors occurred!
      make: *** 没有指明目标并且找不到 makefile。 停止。
      Traceback (most recent call last):
        File "/home/hc-em/miniforge3/envs/isaaclab_work/lib/python3.11/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
          main()
        File "/home/hc-em/miniforge3/envs/isaaclab_work/lib/python3.11/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
          json_out["return_val"] = hook(**hook_input["kwargs"])
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/hc-em/miniforge3/envs/isaaclab_work/lib/python3.11/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 280, in build_wheel
          return _build_backend().build_wheel(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/build_meta.py", line 435, in build_wheel
          return _build(['bdist_wheel', '--dist-info-dir', str(metadata_directory)])
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/build_meta.py", line 423, in _build
          return self._build_with_temp_dir(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/build_meta.py", line 404, in _build_with_temp_dir
          self.run_setup()
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/build_meta.py", line 512, in run_setup
          super().run_setup(setup_script=setup_script)
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup
          exec(code, locals())
        File "<string>", line 50, in <module>
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/__init__.py", line 115, in setup
          return distutils.core.setup(**attrs)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/_distutils/core.py", line 186, in setup
          return run_commands(dist)
                 ^^^^^^^^^^^^^^^^^^
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/_distutils/core.py", line 202, in run_commands
          dist.run_commands()
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/_distutils/dist.py", line 1002, in run_commands
          self.run_command(cmd)
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/dist.py", line 1102, in run_command
          super().run_command(command)
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
          cmd_obj.run()
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/command/bdist_wheel.py", line 370, in run
          self.run_command("build")
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/_distutils/cmd.py", line 357, in run_command
          self.distribution.run_command(command)
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/dist.py", line 1102, in run_command
          super().run_command(command)
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
          cmd_obj.run()
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/_distutils/command/build.py", line 135, in run
          self.run_command(cmd_name)
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/_distutils/cmd.py", line 357, in run_command
          self.distribution.run_command(command)
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/dist.py", line 1102, in run_command
          super().run_command(command)
        File "/tmp/pip-build-env-cw_pa5zi/overlay/lib/python3.11/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
          cmd_obj.run()
        File "<string>", line 26, in run
        File "<string>", line 38, in build_extension
        File "/home/hc-em/miniforge3/envs/isaaclab_work/lib/python3.11/subprocess.py", line 413, in check_call
          raise CalledProcessError(retcode, cmd)
      subprocess.CalledProcessError: Command 'cmake ..; make -j' returned non-zero exit status 2.
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for egl_probe
Successfully built isaaclab_mimic robomimic
Failed to build egl_probe
error: failed-wheel-build-for-install

× Failed to build installable wheels for some pyproject.toml based projects
╰─> egl_probe

```