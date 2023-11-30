import gymnasium as gym
import time
import numpy as np

# from gym.envs.registration import register
import argparse
from gym_multigrid.envs.mapf_env import FlatMultiGridObsWrapper

parser = argparse.ArgumentParser(description=None)
parser.add_argument("-e", "--env", default="mapf", type=str)

args = parser.parse_args()


def main():
    if args.env == "mapf":
        # register(
        #     id="multigrid-mapf-v0",
        #     entry_point="gym_multigrid.envs:MapfEnv",
        # )
        # env = gym.make("multigrid-mapf-v0")

        env = FlatMultiGridObsWrapper(
            gym.make(
                "gym_multigrid:multigrid-mapf-v0",
                scenario_file="/media/project0/MAPF_group/highway_optimization/MAPFCompetition2023/example_problems/warehouse.domain/warehouse_small_10.json",
                edge_weights=np.random.rand(33, 57, 4), # shape [h, w, 4]
                max_steps=10,
                # render_mode="human",
            )
        )
        # env = gym.make(
        #     "gym_multigrid:multigrid-mapf-v0",
        #     scenario_file="/home/vineet/competition/Start-Kit/example_problems/warehouse.domain/warehouse_small_10.json",
        #     max_steps=10,
        #     # render_mode="human",
        # )

    _ = env.reset()

    nb_agents = len(env.agents)

    i = 0
    while True:
        # env.render()
        time.sleep(0.1)

        # ac = [env.action_space.sample() for _ in range(nb_agents)]
        ac = env.action_space.sample()

        print("--------------------------------------------------------")
        print(type(ac))
        print(ac)
        print("--------------------------------------------------------")
        obs, _, terminated, truncated, _ = env.step(ac)

        # print("--------------------------------------------------------")
        # print(type(obs))
        # print(obs[0])
        # print(obs[0].shape)
        # print("--------------------------------------------------------")
        i += 1
        if terminated or truncated:
            print("Num iters ", i)
            break


if __name__ == "__main__":
    main()
