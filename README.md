# Introduction-to-Deep-Learning-From-Logical-Calculus-to-Artificial-Interlligence
对于《深入浅出深度学习：从逻辑运算到人工智能》此书中的学习记录

对前馈神经网络中书上范例的代码进行了一点修改,运行结果放在result.txt文件中






环境配置中的常见问题：

ImportError: DLL load failed while importing _pywrap_tensorflow_internal: 找不到指定的模块。

解决办法：

1.这可能是电脑缺少Visual C++导致的，去这里下载：Visual [C++下载网站](https://learn.microsoft.com/zh-cn/cpp/windows/latest-supported-vc-redist?view=msvc-170)

2.Check your Python version: TensorFlow only supports Python 3.5-3.8. If you're using a different version, you may need to switch to a supported version.检查你的Python版本： TensorFlow只支持Python 3.5-3.8。如果你正在使用不同的版本，你可能需要切换到支持的版本。

3.更新 pip, setuptools, 和 wheel
    pip install --upgrade pip setuptools wheel

4.检查Pyhton安装路径


另外的常见问题：

tensorboard和tensorflow-intel所需要的protobuf包与安装的那个包之间存在版本冲突。
tensorboard要求protobuf的版本大于或等于3.19.6，而tensorflow-intel要求protobuf的版本大于或等于3.20.3但小于5.0.0dev。

解决这个问题，你可以将 protobuf 包升级到与 tensorboard 和 tensorflow-intel 都兼容的版本。
