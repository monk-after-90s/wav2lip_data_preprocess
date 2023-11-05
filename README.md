# wav2lip训练数据预处理综合工具

取代原项目的预处理脚本

## Install

tested on Python 3.7.7

```shell
conda env create -f environment.yml
conda activate wav2lip_data_preprocess
```

## Run

这是个傻瓜化的项目，只需要按数字依次运行脚本即可。更多参数，查看脚本的--help，如：

```shell
python 1_extract_audio.py  --help
```

也可以使用“&&”连接多个命令一键运行全部命令，比如：

```shell
python 1_extract_audio.py && python 2_noise_depress.py && ...
```