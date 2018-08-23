# 利用卷积网络的图像特征点检测

### 计划记录

- 第一阶段：
	- 合成形状数据集的构建
	- 建立Basic Dector检测器
	- 利用Basic Dector生成特征点数据集
- ...
- 小网络的特征点检测

### 研究记录
- 第一阶段：
	- 使用opencv得到了合成数据集
	- Basic Dector基本框架已经建立
		- <image src='net.png' width='400px'>
		- loss function 还没有使用和论文一样的形式
		- 按照原论文，特征点位置只能是某个确定的像素位置，而像SIFT等应该可以是像素点之间的位置
		- 建立从H*W-->(H/8)*(W/8)*65-->`ing`
	- H-Patches的数据集使用  -->`ing`
	- 同型适应方法
- ...

### 文件说明
- `logs`：存放log文件，包括TF的TensorBoard可视化文件
- `basic_model.py`:用于存放模型基类，包括Basic Dector和Super Point
- `magic_point.py`:Basic Dector模型，直接运行会测试Basic Dector
	- 需要在同级目录下创建train(存放特征点npy文件)，training(图像文件)
- `utils.py`:封装的网络块文件
- `test_bins.py`:目前包含测试Basic Dector的时候使用的数据供给函数

### 说明
##### 将创建文件夹用于记录项目进度,同时借机学一下GitHub上的协作开发
- 需求：
	- loss function的研究修改
	- Hpatch的使用
	- 同型适应方法
- 建议先看论文，目前是想复现论文，所以几乎就是按照论文做的
- 网络的话都是Tensorflow搭出来的。另一方面还需要复习一下opencv的图像处理