This is task for JetBrains internship - MountainCar-v0 agent.

It uses DDQN with PER to learn.

[Result on YouTube](https://youtu.be/DWDwCWttz7I)

[Model](https://drive.google.com/file/d/1W94gzB5P_3RjRowVIKlIWxrsKQqqtFB2/view?usp=sharing)  
Unpack it to the root of repository and run demo with `python demonstration.py`   
You need:
* python 3
* pytorch 1.1
* openai gym
* numpy

To train new model run `python train.py` with empty `models` dir in the root. You can tune hyperparameters in `train.py` file.