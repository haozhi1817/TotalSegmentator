<!--
 * @Author: HaoZhi
 * @Date: 2024-07-10 15:46:21
 * @LastEditors: HaoZhi
 * @LastEditTime: 2024-07-10 18:52:01
 * @Description: 
-->
# 写在前面的话
## 重构的理由
* 本代码重构自TotalSegmentator-V1官方代码，由于官方代码中涉及模型参数存放位置，临时文件的保存位置等默认参数，我并不喜欢有人在我不知情的情况下对我的电脑进行操作，即使是新建临时文件夹且最后会删除，我也同样无法忍受。因此选择重构官方代码，同时也期望在重构的过程中对代码有进一步的了解。
## 一致性
* 针对三个测试数据，我可以保证两个测试数据在所有可选参数的情况下，重构代码的结果与官方代码的结果完全一致，但是在一个较大的测试数据上，这个一致性无法保证，且尝试debug并未发现原因，考虑到差异并不大，因此暂时将这个bug留在这里。

# 代码使用
## 官方代码
* 安装
```bash
pip install TotalSegmentator
```
* 使用
```python
from totalsegmentator.python_api import totalsegmentator

totalsegmentator(input_path, output_path, fast = False, ml = True)
# input_path: 输入数据路径，格式为***.nii/nii.gz
# output_path: 输出数据路径，格式为***.nii/***.nii.gz
# fast: 是否使用fast模式，使用，则利用一个模型进行预测，不使用，则利用5个模型预测
# ml: 是否将多个标签融合到一个数据中去，如果设置为False，则output_path应该是文件夹路径，在该文件夹中，每个标签单独保存为一份nii
```

## 重构代码
* 安装
```bash
pip install TotalSegmentator
```
* 下载并解压参数
> 在**https://github.com/wasserth/TotalSegmentator/releases/download**下载对应的参数，具体路径详见官方代码 
> 将所有参数直接加压到某个文件夹中，解压后的目录格式为**root/TaskXXX_TotalSegmentator.../nnUNetTrainerV2_ep4000_nomirror__nnUNetPlansv2.1**
* 使用
```python
infer(nii_input_path,
      model_folder,
      tmp_folder,
      nii_out_path,
      if_fast=False,
      split_margin=20,
      all_in_gpu=True,
      mix_precision=True,)
# nii_input_path: 输入数据路径，格式为***.nii/nii.gz
# model_folder: 模型参数保存路径
# tmp_folder: 临时文件存放路径，程序运行结束后，该路径下的所有内容会被清空
# nii_out_path：输出文件路径，所有标签都会融合到该文件中
# if_fast：使用快速(1个模型)或者慢速(5个模型)
# split_margin：当数据庞大时，处于内存考虑，会将数据切分为3块，每一块之间存在overlap，记为margin，官方代码中设置为20
# all_in_gpu: 是否在gpu上完成计算
# mix_precision： 是否使用混合精度
```

# 代码结构
## load_data
### 流程
1. 加载输入数据，并根据是否使用fast模式，将数据resize到各向同性，其中fast模式下，resize后的spacing为3.0，非fast模式下位1.5
2. 如果数据非常大(数据shape的乘积超过(256\*256\*900)且resize后层面数超过200且处于非fast模式)，则将数据沿着层面数方向等分为3份，第一份向后延伸margin层面，第二份分别向前向后延伸margin层面，第三份向前延伸margin层面。
3. 将上述切分或者无需切分的数据保存到临时文件夹的tmp_input目录下，记为(s01.nii.gz/s02.nii.gz/s03.nii.gz)

### 注意
1. 数据加载过程中会使用nib.as_closest_canonical函数对数据进行调整，具体细节不甚清楚，如果没有这一步操作，最后的预测结果将会是爆炸性的失败。

## build_model
### 流程
1. 根据传入的model保存路径，加载模型初始化文件，并完成初始化。
2. 根据传入的model保存路径，将参数加载到内存中，注意此时并没有将参数加载到模型中。

## nnunet_predict

### 流程
1. 利用多进程，对数据进行预处理(主要内容是resize，crop以及normalize，resize则是进一步保证各向同性，如果在前文的load_data中执行过resize操作，这里不会再执行，crop主要是去除resize插值时可能出现的全0情况，一般也不会出现，这个预处理实际上是NNUnet自带的预处理，由于TotalSegmentator有自己的数据处理方式，所以大多数情况下，这里的数据预处理都没什么作用，但是这里的resize中使用到了batchgenerators这个库里的函数，不确定是否是nnunet必须得)，记录原始数据结构信息，如果数据足够大(shape的乘积大于2e9 / 4 * 0.85)，则会暂时将数据保存为npy格式。
2. 模型加载参数，开始预测，预测结果的输出为argmax结果。
3. 将上述预测结果根据task保存到临时文件夹tmp_pred/task中。

### 说明
1. 数据预处理中的norm是先卡上下阈值然后减均值除方差，这里的上下阈值，均值方差都是预定义好的，保存在下载好的参数中的model.pkl文件里plans.dataset_properties.intensityproperties中，这些数值与实际上的CT窗宽窗位并不一致。

## infer
### 流程
0. 初始化临时文件夹，创建tmp_input, tmp_pred两个子文件夹，并根据task_id在tmp_pred文件夹下创建task子文件夹。
1. 利用build_mode根据task_id初始化模型。在fast模式下，task_id 为 [256], 非fast模式下为[251, 252, 253, 254, 255]。此时整个内存中会有1或者5个初始化模型，后续可以优化这一部分，当我们需要某个模型时，再单独初始化他。
2. 利用load_data对数据进行预处理，将其保存到tmp_input中。
3. 利用nnunet_predict进行数据预测，并将结果保存到tmp_pred中。需要说明的是，如果数据足够大，被切分为3份(分别记为s01, s02, s03)，同时使用5个模型进行预测，nnunet_predict会产生15个结果，根据task，分别保存在5个文件夹内。
4. 对上述结果进行后处理，后处理主要包含3部分：
   1. 标签合并：当使用5个模型预测时，需要将前文的5个task结果根据class_map文件中提供的转换规则进行合并
   2. 数据拼接：当数据被切分为3份后，需要将其拼接位完整数据
   3. 逆变换：将预测结果resize回原始数据的spacing，同时执行as_closest_canonical的逆向操作
5. 清空临时文件夹

# 临时文件夹标准结构
```bash
|---tmp_input
    |---s01.nii.gz
    |---s02.nii.gz
    |---s03.nii.gz
|---tmp_pred
    |---task251
        |---s01.nii.gz
        |---s02.nii.gz
        |---s03.nii.gz
    |---task252
        |---s01.nii.gz
        |---s02.nii.gz
        |---s03.nii.gz
    |---task253
        |---s01.nii.gz
        |---s02.nii.gz
        |---s03.nii.gz
    |---task254
        |---s01.nii.gz
        |---s02.nii.gz
        |---s03.nii.gz
    |---task255
        |---s01.nii.gz
        |---s02.nii.gz
        |---s03.nii.gz
```
# 后续优化
1. 一致性问题
2. 模型依次创建
3. 关闭nnunet打印的各种信息