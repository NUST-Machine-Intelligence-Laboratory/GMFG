# Guided by Meta-set: A Data-driven Method for Fine-Grained Visual Recognition

Introduction
------------
This is the source code for our paper **Guided by Meta-set: A Data-driven Method for Fine-Grained Visual Recognition**

Network Architecture
--------------------
The architecture of our proposed model is as follows
![network](network.png)

Installation
------------
After creating a virtual environment of python 3.7, run `pip install -r requirements.txt` to install all dependencies

How to use
------------
The code is currently tested only on GPU
* **Data Preparation**
    - We test our approach on [WebFG-496][1]. Download data into project root directory and uncompress them using
        ```
        wget https://web-fgvc-496-5089-sh.oss-cn-shanghai.aliyuncs.com/web-bird.tar.gz
        tar -xvf web-bird.tar.gz
      
        # optional
        wget https://web-fgvc-496-5089-sh.oss-cn-shanghai.aliyuncs.com/web-car.tar.gz
        wget https://web-fgvc-496-5089-sh.oss-cn-shanghai.aliyuncs.com/web-aircraft.tar.gz
        tar -xvf web-car.tar.gz
        tar -xvf aircraft-car.tar.gz
        ```
    - Meta sets are created through randomly selecting 10 images for each categroy from benchmark datasets.
    
* **Source Code**

    - If you want to train the whole network from begining using source code on the web fine-grained dataset, please follow subsequent steps
    
      - Choose a dataset, create soft link to dataset by
       ```
       ln -s web-bird bird
      
       # optional
       ln -s web-car car
       ln -s web-aircraft aircraft
       ```

      - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id and `data_base` to proper dataset in  ``` run_train_resnet.sh ```
      
      - Activate virtual environment(e.g. conda) and then run the script ```bash run_train_resnet.sh``` to train a resnet50 model.
     
     
    ![table](performance.png)


  [1]: https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset