## 基于YOLOv8与Qt的多目标跟踪智能交通路况监控系统
# 1.演示视频
https://www.bilibili.com/video/BV1yX4y1m7Fe/?spm_id_from=333.999.0.0<br>
https://youtu.be/_77LrsXaYzM
# 2.安装（INSTALL)
## 第一步（FIRST STEP) 安装ANACONDA<br>
1）访问Anaconda官网：https://www.anaconda.com/products/individual<br>
2）选择相应的操作系统版本并下载对应的安装包（推荐下载64位版本）<br>
3）打开下载的安装包，按照提示进行安装即可<br>
4）创建一个虚拟环境：<br>
conda create --name 自命名 python=3.9.16<br>
<br>
## 第二步（SECOND STEP) pip install -r requirements.txt<br>
激活环境并安装相应的库：  activate 自命名-> pip install -r requirements.txt<br>
这一步会安装cpu版本的torch与torchvision，如果想要更好的帧数体验请安装cuda版本哦，安装cuda版本很简单，首先要有英伟达显卡，其次nvdia-smi查看cuda driver驱动版本号，上英伟达官网选择对应cuda版本号的cuda套件安装，最后去torch官网选择自己安装的cuda套件版本使用conda或者pip安装即可。<br>
# 3.运行
配置好环境后在含有main.py的工作目录下运行main.py即可，也可以下载以下链接里的压缩包使用exe文件运行：<br>
https://pan.baidu.com/s/1U9dskWzOouF4y1_s7KnPfg?pwd=Zlad 提取码: Zlad

## Multi-Object Tracking Intelligent Traffic Monitoring System based on YOLOv8 and Qt

# 1. Demo Video
https://www.bilibili.com/video/BV1yX4y1m7Fe/?spm_id_from=333.999.0.0<br>
https://youtu.be/_77LrsXaYzM

# 2. Installation
## First Step: Install Anaconda
1) Visit the Anaconda official website: https://www.anaconda.com/products/individual<br>
2) Select the appropriate operating system version and download the corresponding installation package (it is recommended to download the 64-bit version)<br>
3) Open the downloaded installation package and follow the prompts to install it<br>
4) Create a virtual environment:<br>
conda create --name your_env_name python=3.19.16<br>

## Second Step: pip install -r requirements.txt
Activate the environment and install the required libraries:<br>
activate your_env_name -> pip install -r requirements.txt<br>

This step will install the CPU version of torch and torchvision. If you want a better frame rate experience, please install the CUDA version. Installing the CUDA version is very simple. First, you need an NVIDIA graphics card. Next, use nvidia-smi to check the version number of the CUDA driver. Then, select the corresponding CUDA package version according to the CUDA driver version number on the NVIDIA official website, and finally, choose the CUDA package version you installed and use conda or pip to install it on the torch official website.

# 3. Running
After configuring the environment, run main.py in the working directory that contains the file. You can also download the compressed package from the following link and use the exe file to run it:<br>
https://pan.baidu.com/s/1U9dskWzOouF4y1_s7KnPfg?pwd=Zlad Password: Zlad

## 参考(REFERENCE)
https://github.com/Jai-wei/YOLOv8-PySide6-GUI<br>
https://github.com/ultralytics/ultralytics<br>
https://doc.qt.io/qtforpython-6/






