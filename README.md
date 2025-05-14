# GRACE_pensieve_ppo

## Install the dependencies

**Note: you need to have `git` and `conda` before installation**
```bash
# clone the repo
git clone https://github.com/lumingkai234/Grace_pensieve_ppo # clone the repo
cd Grace

# install the dependencies
sudo apt install ffmpeg ninja-build # install the ffmpeg and ninja
conda env create -f env.yml # creating the conda environment
```

**Note: you may need to verify the PyTorch installation and reinstall it yourself if there are any PyTorch-related errors**




## Download the model and the test videos

The testing videos can be downloaded at: https://drive.google.com/file/d/1iQhTfb7Kew_z97kDVoj2vOmQqaNjBK9J/view?usp=sharing

The models for Grace can be downloaded at: https://drive.google.com/file/d/1IWD-VUc0RPXXhBzoH5j9YD6bl8kzYYJ1/view?usp=sharing

```bash
# download the models
cp /your/download/path/grace_models.tar
cd models/
tar xzvf grace_models.tar 

# download and extract the videos
cd ../videos/
cp /your/download/path/GraceVideos.zip .
unzip GraceVideos.zip
```

## About Grace_pensieve_ppo
带宽的数据集在./cooked_traces下面 

grace_pensieve_ppo_model.py文件是ppo模型的网络结构的定义文件

grace_pensieve_ppo_basic.py文件是ppo类：涉及动作的选择逻辑、网络的更新、模型的加载和保存

grace_pensieve_ppo_replaymemory.py文件是经验回放的空间的定义

grace_pensieve_ppo_env.py文件是ppo和环境交互的逻辑：涉及到trace的读取、丢包的计算、state的更新

grace_pensieve_ppo_train.py文件是主函数  包括训练逻辑和对模型在验证集上的验证

grace_pensieve_ppo_test.py是测试ppo模型在测试集上的效果



