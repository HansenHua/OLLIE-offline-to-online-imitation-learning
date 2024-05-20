# OLLIE
This work "OLLIE: Imitation Learning from Offline Pretraining to Online Finetuning" has been accepted by ICML'24.
## :page_facing_up: Description
we propose a principled offline-to-online IL method, named \texttt{OLLIE}, that simultaneously learns a near-expert policy initialization along with an \textit{aligned discriminator initialization}, which can be seamlessly integrated into online IL, achieving smooth and fast finetuning. Empirically, \texttt{OLLIE} consistently and significantly outperforms the baseline methods in \textbf{20} challenging tasks, from continuous control to vision-based domains, in terms of performance, demonstration efficiency, and convergence speed. This work may serve as a foundation for further exploration of pretraining and finetuning in the context of IL.
## :wrench: Dependencies
- Python == 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch == 1.8.1](https://pytorch.org/)
- [MuJoCo == 2.3.6](http://www.mujoco.org) 
- NVIDIA GPU (RTX A6000) + [CUDA 11.1](https://developer.nvidia.com/cuda-downloads)
### Installation
1. Clone repo
    ```bash
    git clone [https://github.com/HansenHua/OLLIE-offline-to-online-imitation-learning.git](https://github.com/HansenHua/OLLIE-offline-to-online-imitation-learning.git)
    cd MFPO-Online-Federated-Reinforcement-Learning
    ```
2. Install dependent packages
    ```
    pip install -r requirement.txt
    ```
## :zap: Quick Inference

Get the usage information of the project
```bash
cd code
python main.py -h
```

## :computer: Training

We provide complete training codes for OLLIE.<br>
You could adapt it to your own needs.

	```
    python main.py CartPole-v1 MFPO train
	```
	The log files will be stored in [https://github.com/HansenHua/OLLIE-offline-to-online-imitation-learning](https://github.com/HansenHua/OLLIE-offline-to-online-imitation-learning/tree/main/performance).
## :checkered_flag: Testing
Illustration

We alse provide the performance of our model. The illustration videos are stored in [https://github.com/HansenHua/OLLIE-offline-to-online-imitation-learning/performance](https://github.com/HansenHua/OLLIE-offline-to-online-imitation-learning/tree/main/performance).

## :e-mail: Contact

If you have any question, please email `xingyuanhua@bit.edu.cn`.
