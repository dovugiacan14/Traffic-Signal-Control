# **Reinforcement Learning for Traffic Signal Control**
![image](./assets/resized_image.png)

## Abstract 
Traffic Signal Control (TSC) plays an important role in regulating the flow of traffic at intersections, especially during peak hours. Recent studies have focused on developing decision-making systems to adjust traffic signal lights based on real-time traffic flow monitoring. However, they require complex designs and come with associated costs and manpower. Therefore, we propose applying reinforcement learning for automatic trafic signal adjusments that are suitable for real world traffic conditions. Our experiment results show the improvement for TSC problem on simulated maps. 

## SUMO
In our experiment, we use [OpenAI Gym](https://gymnasium.farama.org/index.html) Interface to deploy Reinforcement Learning algorithm and [SUMO](https://sumo.dlr.de/docs/index.html) simulator serves as an enviroment for operarting and evaluating the effectiveness of Reinforcement Learning controllers in Traffic Signal Control problem. 

Sumo-RL provides a simple interface to instantiate Reinforcement Learning (RL) enviroments with SUMO for Traffic Signal Control.

## Requirements Installation 
### Install Conda 
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

### Create Cona Environment 
```bash
conda create -n tsc python=3.12 
conda activate tsc  
```

### Install Pytorch 
```bash 
conda install pytorch pytorch-cuda -c pytorch -c nvidia
```
### Install Latest SUMO 
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

Set ```SUMO_HOME``` variable (default sumo installation path is /usr/share/sumo)
```bash
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```
**Important:** for a huge performance boost (~8x) with Libsumo, you can declare the variable: 
```bash
export LIBSUMO_AS_TRACI = 1
```

### Install Other Requirements 
```bash
pip install -r requirements.txt
```

## Quick Examples
```bash
python inference.py
```

## Notice 
This is the initial version of our code. We will update the code and add more examples in the future. If you have any questions, please feel free to contact us.

