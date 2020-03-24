# CARLCS-CNN

## Dependency
> Tested in Ubuntu 16.04
* Python 2.7-3.6
* Keras 2.0.0 or newer
* Tensorflow or Theano 0.8.0~0.9.1


## Usage

   ### Data Preparation
  The `/data` folder provides a small sample dataset for quick running. 
  To train and evaluate our model:
  
  1) Download and unzip real dataset from [Google Drive](https://drive.google.com/drive/folders/1GZYLT_lzhlVczXjD6dgwVUvDDPHMB6L7?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1U_MtFXqq0C-Qh8WUFAWGvg) for Chinese users.
  
  2) Replace each file in the `/data` folder with the corresponding real file. 
  
   ### Configuration
   
   Edit hyper-parameters and settings in `config.py`
   
   ### Train and Evaluate
   
   ```bash
   python main.py --mode train
   
   ```bash
   python main.py --mode eval
