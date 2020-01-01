# 论文 Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction代码重构

本代码使用Tensorflow 1.5 重构自项目：https://github.com/lucktroy/DeepST

博客：https://blog.csdn.net/The_lastest/article/details/85001601



## 更新记录

**2020-01-01**

- 在主页增加使用说明

**2019-07-06**

- 修复数据预处理中`reshape`与``transpose`误用的错误

**2018-12-14**

- 完成了整个框架的搭建和调试工作；
- 完成了一次完整的训练和结果记录；

## 使用指南

- **1.安装**

  安装好`tensorflow-gpu 1.5.0`版本，下载好数据集解压后放到`data/TaxiBJ/`目录下即可；

  下载页面：

  #### [TaxiBJ: InFlow/OutFlow, Meteorology and Holidays at Beijing](./data/TaxiBJ/README.md)

  - BJ16_M32x32_T30_InOut.h5
  - BJ15_M32x32_T30_InOut.h5
  - BJ14_M32x32_T30_InOut.h5
  - BJ13_M32x32_T30_InOut.h5
  - BJ_Meteorology.h5
  - BJ_Holiday.txt

  将下载好的这六个文件放到`data/TaxiBJ/`目录

- **2.训练**

  下载完成数据后进入`TaxiBJ`文件夹，在``exptTaxiBJ.py`中设置好相关参数，运行 `python exptTaxiBJ.py`即可开始训练；

  