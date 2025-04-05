# lux-ai-s3-ppo
In this project, I implement a PPO algorithm in JAX for the [LuxAI S3](https://www.kaggle.com/competitions/lux-ai-season-3) Kaggle Competition. The main goal of this project was for me to get familiar with Reinforcement Learning and working with JAX. Therefore, I decided to focus more on understanding the algorithm and the implementation details, less on other factors such as hyperparameter tuning and model development.

I used code from [purejaxrl](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py) to help me make this repository

# Running The Code
To run this project, you are expected to build a Docker image from a Dockerfile and launch an interactive container from which you will run all the scripts. The only items you need to install locally is Docker and NVIDIA GPU Drivers, the rest will be installed via the image.

First clone the repository to your local directory. Then go into the repository folder and build the Docker image using the following command:

```
sudo docker build -t lux-ai-s3-ppo .
```

After the Docker Image has been built you can create an interactive container using the following command:

```
sudo docker run --name lux-ai-s3-ppo-container --gpus all -it -p 23:22 lux-ai-s3-ppo &
```
**Note**: The port tag is included incase you might wish to ssh into the container. It is recommended to work with Docker containers using an IDE such as Visual Studio Code.

Inorder to run PPO training for our given environment and model, run the following command:

```
python3 src/ppo.py
```
If you accidentally exited the container and need you access the container shell again run the command:

```
sudo docker exec -it lux-ai-s3-ppo-container /bin/bash
```
