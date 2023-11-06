# Multi-Agent Gridworld Environment (MultiGrid)

## Installation
How to run:

clone and cd into the repo and run the following command:
```
pip install -e .
```

specify the path to the competition .json files in `mapf_env_test.py`
the environment will load the map, agent and task files from the relative path provided in the .json

```
python3 mapf_env_test.py
```
----------------------------

The repo for maps can be found [here](https://github.com/MAPF-Competition/Start-Kit/tree/main)

----------------------------

Lightweight multi-agent gridworld Gym environment built on the [MiniGrid environment](https://github.com/maximecb/gym-minigrid). 

Requirements:
- Python 3.5+
- OpenAI Gym
- NumPy
- Matplotlib

## Design

The environment can be either fully or partially observable. Each grid cell is encoded with a tuple containing:
- The type of the object (can be another agent)
  - Provided object types are: wall, floor, lava, door, key, ball, box, goal, object goal and agent
- The color of the object or other agent
- The type of the object that the other agent is carrying
- The color of the object that the other agent is carrying
- The direction of the other agent 
- Whether the other agent is actually one-self (useful for fully observable view)

Actions in the basic environment:
- Turn left
- Turn right
- Move forward
- Done (task completed, optional)

