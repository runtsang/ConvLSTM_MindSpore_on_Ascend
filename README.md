# 目录

# 模型名称

> 模型: ConvLSTM
>
> 论文链接: https://arxiv.org/pdf/1506.04214.pdf
>
> 简介:
>
> ​	模型ConvLSTM作者通过实验**证明了ConvLSTM在获取时空关系上比LSTM有更好的效果。** ConvLSTM不仅可以预测天气，还能够解决其他时空序列的预测问题。比如视频分类，动作识别等。此次数据集为[Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/)

## 模型架构

> ![image-20220727091656901](https://user-images.githubusercontent.com/70456146/181144611-67b2a5ef-6b7e-4310-aaa1-fd29308cd054.png)

## 数据集

> 训练数据集: [The MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/)
>
> 验证精度数据集: [Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/)
>
> 下载数据集请通过目录下的脚本下载
>
> ```
> source ./download.sh
> ```

## 写在前面

本仓库为模型在openi启智训练运行版本，需依靠启智平台算力，硬件平台为Ascend 910。

若需要将此项目运行在GPU上，请移步此仓库：https://github.com/zRAINj/ConvLSTM_MindSpore

## 训练与评估过程

> 训练过程采用MNIST手写数字数据库，其中有60,000个示例的训练集和10,000个示例的测验集。它是MNIST的子集。数字已被归一化并以固定大小的图像为中心。训练及测验过程中通过动态生成视频数据来进行训练。特别需要注意，训练过程中生成的数字数量为3，相较于评估中的2个数字更多。

### 训练

此次训练采用openi平台云脑训练任务，openi仓库：https://git.openi.org.cn/zrainj/ConvLSTM_MindSpore

具体云脑训练配置如下：（过低可能会导致训练速度过慢）

> 计算资源：Ascend NPU
>
> AI引擎：
> – MindSpore version : 1.5.1
> – Python version : 3.7
> – OS platform and distribution : euleros2.8-aarch64
>
> 规格：
>
>  NPU: 1*Ascend 910, CPU: 24, 显存: 32GB, 内存: 256GB
>
> 启动文件： train.py

### 评估过程

> 评估过程采用[Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/)作为测试集。



------



## 性能

### 训练性能

| train_loss | valid_loss | SSIM     | MAE        | MSE       |
| ---------- | ---------- | -------- | ---------- | --------- |
| 0.000976   | 0.000961   | 0.777687 | 221.285598 | 94.498799 |

### 评估性能

| test_loss | SSIM     | MAE        | MSE       |
| --------- | -------- | ---------- | --------- |
| 0.000638  | 0.833904 | 156.482312 | 62.759463 |

## 随机情况说明

> 载入权重模型后继续训练会有较大精度浮动

## 参考模板

[ConvLSTM-PyTorch](https://github.com/jhhuang96/ConvLSTM-PyTorch)

## 贡献者

* [曾润佳](https://github.com/zRAINj) (广东工业大学)

## ModelZoo 主页

请浏览官方[主页](https://gitee.com/mindspore/models)。