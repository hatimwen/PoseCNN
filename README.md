# PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes

Created by Yu Xiang at [RSE-Lab](http://rse-lab.cs.washington.edu/) at University of Washington and NVIDIA Research.

### Introduction

We introduce PoseCNN, a new Convolutional Neural Network for 6D object pose estimation. PoseCNN estimates the 3D translation of an object by localizing its center in the image and predicting its distance from the camera. The 3D rotation of the object is estimated by regressing to a quaternion representation. [arXiv](https://arxiv.org/abs/1711.00199), [Project](https://rse-lab.cs.washington.edu/projects/posecnn/)

[![PoseCNN](http://yuxng.github.io/PoseCNN.png)](https://youtu.be/ih0cCTxO96Y)

### License

PoseCNN is released under the MIT License (refer to the LICENSE file for details).

### Citation

If you find PoseCNN useful in your research, please consider citing:

    @inproceedings{xiang2018posecnn,
        Author = {Xiang, Yu and Schmidt, Tanner and Narayanan, Venkatraman and Fox, Dieter},
        Title = {PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes},
        Journal   = {Robotics: Science and Systems (RSS)},
        Year = {2018}
    }

 
### Environment

本人系统环境：
- Ubuntu 16.04
- Tensorflow 1.8（from source）
- Python 2.7
- Cuda 10.0 & cuddn 7.3.1

### Setting

***
#### <center> 1.搭建虚拟环境 </center>
第一步，创建专属于PoseCNN的虚拟环境，之后install的包都在此虚拟环境中。
虚拟环境的好处不用多说了吧，反正对Ubuntu系统的折腾越少越好！！！
我用 conda 创建的环境：
- <code> conda create -n posecnn python=2.7 </code>
激活环境：
- <code> conda activate posecnn </code>
如果不用这个环境，记得deactivate：
- <code> conda deactivate posecnn </code>

***
#### <center> 2.pip install </center>
- <code> pip install opencv-python </code>

如果不行试一下：<code> sudo apt-get install libopencv-dev </code>

- <code> pip install mock enum34</code>
- <code> pip install matplotlib numpy keras Cython Pillow easydict transforms3d </code>
- <code> pip install OpenEXR </code>
- <code> sudo apt-get install libsuitesparse-dev libopenexr-dev metis libmetis-dev </code>

***
#### <center> 3.TensorFlow </center>
注意一定要从源码安装，虽然很繁琐，但是经过实践证明，pip install安装出来的TensorFlow不好用。。
此外，使用gcc 4.8和g++ 4.8对后续的依赖包进行编译。
> 
> - <code> sudo apt-get install gcc-4.8 </code>
> - <code> sudo apt-get install g++-4.8 </code>
> - <code> sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 10 </code>
> - <code> sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 30 </code>
> - <code> sudo update-alternatives --config gcc </code> 输入选择 1
> 
> - <code> sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 10 </code>
> - <code> sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 30 </code>
> - <code> sudo update-alternatives --config g++ </code> 输入选择 1
> 
> 测试一下gcc和g++的版本，显示4.8就更换完毕了:
> 
> - <code> gcc --version </code>
> 
> - <code> g++ --version </code>

接下来安装bazel，并选择0.10.0版本，本文选择下载sh文件进行安装，
> 下载地址：[https://github.com/bazelbuild/bazel/releases/download/0.10.0/bazel-0.10.0-installer-linux-x86_64.sh](https://github.com/bazelbuild/bazel/releases/download/0.10.0/bazel-0.10.0-installer-linux-x86_64.sh)
> 下载好之后，安装：
> - <code> chmod +x bazel-0.10.0-installer-linux-x86_64.sh </code> 修改文件权限
> - <code> ./bazel-0.10.0-installer-linux-x86_64.sh --user </code> 进行安装
> 接着添加环境变量：
> - <code> gedit ~/.bashrc </code>
> - <code> export PATH="\$PATH:\$HOME/bin" </code>

下面下载安装TensorFlow：
> - `git clone https://github.com/tensorflow/tensorflow.git`
> - `cd tensorflow`
> - `git checkout r1.8`
> - `./configure`
> 这一步，配置文件会问很多问题，对应回答y/n即可:
> 
>> 注意 Python 及其sitepackage的路径要与你之后环境路径相对应
>> 比如我在posecnn虚拟环境中运行的话，我的python路径就是 .../.conda/env/posecnn/bin/python
> 大部分都选择n，但是询问cuda时，要根据你的电脑实际选择
> 
> 然后编译源文件：
> - `bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package`
> 生成安装包：
> - `bazel-bin/tensorflow/tools/pip_package/build_pip_package  ~/software/tensorflow`
> 最后安装：
> - `pip install /tmp/tensorflow_pkg/tensorflow-1.8.0-cp27-cp27mu-linux_x86_64.whl`
> 至此，TensorFlow的源码安装大功告成，可以import测试一下。

***
#### <center> 4.Eigen </center>

```
wget https://bitbucket.org/eigen/eigen/get/3.3.0.zip
# 提取解压压缩包
# 重命名文件夹为eigen
cd eigen
mkdir build && cd build
cmake ..
make
sudo make install
```
***
#### <center> 5.Nanoflann </center>

```
wget https://github.com/jlblancoc/nanoflann/archive/ad7547f4e6beb1cdb3e360912fd2e352ef959465.zip
# 提取解压压缩包
# 重命名文件夹为nanoflann
sudo apt-get install build-essential cmake libgtest-dev
cd nanoflann
mkdir build && cd build && cmake ..
make && make test
sudo make install
```
***
#### <center> 6.Pangolin </center>

```
wget https://github.com/stevenlovegrove/Pangolin/archive/1ec721d59ff6b799b9c24b8817f3b7ad2c929b83.zip
# 提取解压压缩包
# 重命名文件夹为Pangolin
cd Pangolin
# Add folowing line to the CMakeLists.txt:
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
mkdir build
cd build
cmake ..
cmake --build .
```
***
#### <center> 7.Boost </center>

```
wget https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.bz2
# 提取解压压缩包
# 重命名文件夹为boost
cd boost
./bootstrap.sh
sudo ./b2
sudo ./b2 install
```
***
#### <center> 8.Sophus </center>

```
wget https://github.com/strasdat/Sophus/archive/ceb6380a1584b300e687feeeea8799353d48859f.zip
# 提取解压压缩包
# 重命名文件夹为Sophus
cd Sophus
mkdir build && cd build
cmake ..
make
sudo make install
```
***
#### <center> 9.NLOPT </center>

```
wget https://github.com/stevengj/nlopt/archive/74e647b667f7c4500cdb4f37653e59c29deb9ee2.zip
# 提取解压压缩包
# 重命名文件夹为nlopt
cd nlopt
mkdir build
cd build
cmake ..
make
sudo make install
```
至此，所有依赖包配置完毕，下面针对源代码进行编译运行。
***
#### <center> 10.Compile lib/kinect_fusion </center>
先注释掉/usr/local/cuda/include/crt/common_functions.h的第75行
 ` #define __CUDACC_VER__ "__CUDACC_VER__ is no longer supported.  Use __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, and __CUDACC_VER_BUILD__ instead." `
因为这个[issue](https://github.com/BVLC/caffe/issues/5994)
要是只读权限无法修改，就用`sudo chmod 777 /usr/local/cuda/include/crt/common_functions.h`修改一下权限。
```
cd kinect_fusion
mkdir build
cd build
cmake ..
make
```
编译完记得取消注释刚刚的common_functions.h第75行
***
#### <center> 11.Compile lib/synthesize </center>

```
cd ..
cd ..
cd synthesize
mkdir build
cd build
cmake ..
make
```
Compile the new layers under \$ROOT/lib we introduce in PoseCNN. 
**（注意下面的\$ROOT要换成你实际的PoseCNN代码路径！！！）**
```
cd $ROOT/lib
sh make.sh
```

- run python setup: `python setup.py build_ext --inplace`

- Add pythonpaths

- Add the path of the built libary libsynthesizer.so to python path
```Shell
export PYTHONPATH=$PYTHONPATH:$ROOT/lib:$ROOT/lib/synthesize/build
```

***
#### <center> 12.下载数据集 </center>
- 下载[YCB-Video数据集](https://pan.baidu.com/s/1FG7_wrNbBdFcJmh1UxuDFg)，提取码52xx，解压生成：PoseCNN/data/YCB
- 下载[SUN2012数据集](https://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz),解压生成：PoseCNN/data/SUN2012/data
- 下载[ObjectNet3D数据集](ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_images.zip),解压生成：PoseCNN/data/ObjectNet3D/data

至此，环境配置完毕。接下来直接贴出原作者步骤：

### Running the demo
1. Download our trained model on the YCB-Video dataset from [here](https://drive.google.com/file/d/1UNJ56Za6--bHGgD3lbteZtXLC2E-liWz/view?usp=sharing), and save it to $ROOT/data/demo_models.

2. run the following script
    ```Shell
    ./experiments/scripts/demo.sh # 默认用0号GPU运行!
    # 或者
    ./experiments/scripts/demo.sh --gpuid 1 # 指定1号（也可以选择你喜欢的GPU）运行空格很重要!
    ```

### Running on the YCB-Video dataset
1. Download the YCB-Video dataset from [here](https://rse-lab.cs.washington.edu/projects/posecnn/).数据集上一步已经下好了，这一步不用管~

2. Create a symlink for the YCB-Video dataset (the name LOV is due to legacy, Learning Objects from Videos)
建立软连接，让代码知道你数据集放哪了。
    ```Shell
    cd $ROOT/data/LOV
    ln -s $ycb_data data
    ln -s $ycb_models models
    ```

3. Training and testing on the YCB-Video dataset
    ```Shell
    cd $ROOT

    # training
    ./experiments/scripts/lov_color_2d_train.sh $GPU_ID

    # testing
    ./experiments/scripts/lov_color_2d_test.sh $GPU_ID

    ```

更多可以看下面的参考链接，很详细。更多多的希望通读代码！通读代码！通读代码！

----------
参考：
- [PoseCNN RSE-Lab](https://rse-lab.cs.washington.edu/projects/posecnn/)，RSE-Lab
- [PoseCNN GitHub代码](https://github.com/yuxng/PoseCNN)，yuxng
- [YCB-Video数据集下载](https://pan.baidu.com/s/1FG7_wrNbBdFcJmh1UxuDFg)，提取码52xx，wangg12
- [PoseCNN代码实现大纲](https://github.com/Kaju-Bubanja/PoseCNN)，Kaju-Bubanja
- [PoseCNN代码实现详细](https://github.com/yuxng/PoseCNN/issues/76#issuecomment-452700651)，Luedeke
- [《论文笔记——PoseCNN》](https://blog.csdn.net/nwu_NBL/java/article/details/83176353)，XJTU_Bugdragon
