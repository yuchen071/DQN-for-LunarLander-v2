# DQN for LunarLander v2
 Implementation of reinforcement learning algorithms for the OpenAI Gym environment LunarLander-v2 

## Demo clips
![demo](https://github.com/yuchen071/DQN-for-LunarLander-v2/blob/main/.readme_docs/demo.gif)

## Dependencies
```
gym==0.21.0
imageio==2.13.5
matplotlib==3.5.1
numpy==1.22.0
Pillow==9.0.1
torch==1.10.1+cu102
tqdm==4.62.3
```

## How to use
Training instructions are included in the Jupyter notebook.

## Testing Results
### (a) Loss curve
![loss](https://github.com/yuchen071/DQN-for-LunarLander-v2/blob/main/.readme_docs/loss_curve.png)

### (b) Tune Parameters
#### Target score

I implemented an early stop function which, when the average of the last 100 scores reach a target value, will stop the training process and plot the result. I find that the target score of 250 usually produces more consistent landings and higher total rewards.  
Setting the target score to a lower value like 200, will result in more misses in the final demo. Setting the target score too high, however, will sometimes result in the average score never reaching the target value, which takes more time to train and will not necessarily produce a better result.

| Target=200, gamma=0.99 | Target=250, gamma=0.99 |
|:--:|:--:|
|![t200](https://github.com/yuchen071/DQN-for-LunarLander-v2/blob/main/.readme_docs/target200.gif)|![t250](https://github.com/yuchen071/DQN-for-LunarLander-v2/blob/main/.readme_docs/target250.gif)|  

#### Discount factor

The discount factor ![gamma](https://render.githubusercontent.com/render/math?math=\gamma) determines the importance of future rewards, and the value should be ![0g1](https://render.githubusercontent.com/render/math?math=0\le\gamma%26lt%3B1).  
Setting it too low will make it "short-sighted", only consider rewards nearest to its state. Setting it higher will make it consider more long-term rewards.  
If the discount factor is set equal to or greater than 1, it may cause ![Vpi](https://render.githubusercontent.com/render/math?math=V_\pi) and ![Qpi](https://render.githubusercontent.com/render/math?math=Q_\pi) to diverge.

Experiments as below:

|| Target=230, gamma=0.9 | Target=230, gamma=1.3 |
|:--|:--:|:--:|
|Training Time| 2:02:58, 5000 episodes | 12:09, 5000 episodes |
|Training curve|![g09_curve](https://github.com/yuchen071/DQN-for-LunarLander-v2/blob/main/.readme_docs/gamma09_loss.png)|![g13_curve](https://github.com/yuchen071/DQN-for-LunarLander-v2/blob/main/.readme_docs/gamma13_loss.png)|  
|Result| ![g09_gif](https://github.com/yuchen071/DQN-for-LunarLander-v2/blob/main/.readme_docs/gamma09.gif) | ![g13_gif](https://github.com/yuchen071/DQN-for-LunarLander-v2/blob/main/.readme_docs/gamma13.gif) |

We can see that, ![gamma](https://render.githubusercontent.com/render/math?math=\gamma) set to suboptimal values will result in slow training time and no convergence, and cause the ship to continue hovering and not land.  
![gamma](https://render.githubusercontent.com/render/math?math=\gamma) set to greater than 1 will also cause it to fail to converge, and the ship only fired one side rocket and flew out of control.
