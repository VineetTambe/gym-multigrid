import gym
import time
from gym.envs.registration import register
import argparse

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
        env = gym.make(
            "gym_multigrid:multigrid-mapf-v0",
            # map_file_path="gym_multigrid/envs/maps/mapf_10x10_2.txt",
            # agent_file_path="gym_multigrid/envs/maps/agents_10x10_2.txt",
            # task_file_path="gym_multigrid/envs/maps/tasks_10x10_2.txt",
            # scenario_file="/city.domain/paris_200.json",
            scenario_file="/home/vineet/competition/Start-Kit/example_problems/warehouse.domain/warehouse_small_10.json",
        )

    _ = env.reset()

    nb_agents = len(env.agents)

    while True:
        env.render(mode="human", highlight=True)
        time.sleep(0.1)

        ac = [env.action_space.sample() for _ in range(nb_agents)]

        obs, _, done, _ = env.step(ac)

        if done:
            break


if __name__ == "__main__":
    main()
