from gym_multigrid.multigrid import *
import json
import numpy as np
import matplotlib.pyplot as plt


class MapfEnv(MultiGridEnv):
    """
    Environment in which the agents have to fetch the balls and drop them in their respective goals
    """

    def __init__(
        self,
        # size=10,
        # view_size=3,
        # width=None,
        # height=None,
        # goal_pst=[],
        # goal_index=[],
        # num_balls=[],
        # agents_index=[],
        # balls_index=[],
        # zero_sum=False,
        agent_view_size=3,
        # map_file_path=None,
        # agent_file_path=None,
        # task_file_path=None,
        scenario_file=None,
    ):
        self.num_agents = None
        self.size = None
        self.zero_sum = False

        self.map_file_path = None
        self.agent_file_path = None
        self.task_file_path = None
        self.scenario_file = None
        self.num_rows = None
        self.num_cols = None
        self.loaded_map = None

        self.world = World

        self.AGEGNT_COLOR_IDX = 0
        self.GOAL_COLOR_IDX = 1

        self.goal_locations = []

        if scenario_file is None:
            print("Loading default params! Scenario file path is None")
        else:
            self.scenario_file = scenario_file
            with open(self.scenario_file, "r") as file:
                # Read the contents of the file
                scenario_file_data = file.read()

            # Parse the JSON data into a Python object
            data = json.loads(scenario_file_data)

            base_file_path = self.scenario_file[: self.scenario_file.rfind("/") + 1]

            self.agent_file_path = base_file_path + data["agentFile"]
            self.map_file_path = base_file_path + data["mapFile"]
            self.task_file_path = base_file_path + data["taskFile"]
            self.num_agents = int(data["teamSize"])

        if self.map_file_path is None:
            self.size = 10
            self.map_rows = self.size
            self.map_cols = self.size
            # self.agents_index = agents_index
            print("Loading default params! Map file path is None")
        else:
            self.num_cols, self.num_rows, self.loaded_map = self.load_map_from_file(
                self.map_file_path
            )

            # print(self.num_cols, self.num_rows, self.loaded_map.shape)

            # fig, ax = plt.subplots()

            # # Display the map data
            # ax.imshow(self.loaded_map, cmap="gray", origin="lower")

            # # Add labels and title
            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            # ax.set_title("Occupancy Grid Map")

            # # Show the plot
            # plt.show()

        if self.agent_file_path is None:
            print("Loading default params! Agent file path is None")
            self.num_agents = 10
            agents = []
            for i in range(self.num_agents):
                # TODO add agent start position to the Agent class
                agents.append(
                    Agent(self.world, self.AGEGNT_COLOR_IDX, view_size=agent_view_size)
                )
        else:
            agents = []
            for i in range(self.num_agents):
                agents.append(
                    Agent(self.world, self.AGEGNT_COLOR_IDX, view_size=agent_view_size)
                )

        if self.task_file_path is None:
            print("Loading default params! Task file path is None")

        self.world = World

        super().__init__(
            grid_size=self.size,
            width=self.num_rows,
            height=self.num_cols,
            max_steps=10,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=agent_view_size,
        )

    def load_map_from_file(self, file_path):
        matrix = []
        with open(file_path, "r") as file:
            cnt = 0
            for line in file:
                if cnt < 4:
                    cnt += 1
                    # print(line)
                    if line.split()[0] == "height":
                        height = int(line.split()[1])
                    if line.split()[0] == "width":
                        width = int(line.split()[1])
                    continue
                line = line.strip()
                row = [ord(char) for char in line]
                matrix.append(list(row))
        ret = np.array(matrix)
        ret[ret >= 55] = 255
        ret[ret < 55] = 0

        # ret = 255 - ret
        ret = np.flipud(ret)
        # ret = np.pad(ret, pad_width=1, constant_values=255)
        # add +1 to heigh and width because of padding
        # return width + 1, height + 1, ret
        return width, height, ret

    def deserialize_coords(self, idx, num_cols):
        # row = idx // num_cols + 1  # +1 done here because of padding
        # col = idx % num_cols + 1  # +1 done here because of padding

        row = idx // num_cols  # +1 done here because of padding
        col = idx % num_cols  # +1 done here because of padding

        return row, col

    def serialize_coords(self, row, col, num_cols):
        return row * num_cols + col

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        max_idx = self.width * self.height

        # Load the map
        if self.map_file_path is None:
            # Generate the surrounding walls
            self.grid.horz_wall(self.world, 0, 0)
            self.grid.horz_wall(self.world, 0, height - 1)
            self.grid.vert_wall(self.world, 0, 0)
            self.grid.vert_wall(self.world, width - 1, 0)

        else:
            # pass
            # TODO generate grid as per the map
            # print(f"{self.num_cols=}")
            # print(f"{self.num_rows=}")
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    if self.loaded_map[i, j] == 255:
                        # obstacle
                        self.grid.set(i, j, Wall(self.world))

        print("Done creating map!")

        # Load the agent start locations
        if self.agent_file_path is None:
            # Randomize the player start position and orientation
            for a in self.agents:
                self.place_agent(a)
        else:
            # TODO load agent start position from file
            # pass
            agent_idx = 0
            with open(self.agent_file_path, "r") as file:
                for line in file:
                    # print(int(line.split()[0]), "line number = ", agent_idx)
                    idx = int(line.split()[0])
                    if idx >= max_idx:
                        self.place_agent(self.agents[agent_idx])
                    else:
                        i, j = self.deserialize_coords(idx, self.num_cols)
                        # self.place_agent(self.agents[agent_idx], top=[row, col])
                        # print(agent_idx, idx, len(self.agents))
                        self.put_obj(self.agents[agent_idx], i, j)

                        self.agents[agent_idx].pos = np.array([i, j])
                        self.agents[agent_idx].init_pos = self.agents[agent_idx].pos

                        self.agents[agent_idx].dir = self._rand_int(0, 4)

                        self.agents[agent_idx].init_dir = self.agents[agent_idx].dir

                    agent_idx += 1

                    if agent_idx >= len(self.agents):
                        break
        print("Done adding agents!")

        # Load the agent goal locations
        if self.task_file_path is None:
            for i in range(len(self.agents)):
                pos = self.place_obj(
                    Goal(self.world, self.GOAL_COLOR_IDX),
                    size=[1, 1],
                )
                self.goal_locations.append(pos)
        else:
            with open(self.task_file_path, "r") as file:
                agent_idx = 0
                for line in file:
                    idx = int(line.split()[0])
                    pos = None
                    if idx >= max_idx:
                        # continue
                        pos = self.place_obj(
                            Goal(self.world, self.GOAL_COLOR_IDX),
                            max_tries=100,
                        )
                        if pos is None:
                            print("---------------------------------")
                            print("pos in None!")
                            print("---------------------------------")
                            continue
                    else:
                        i, j = self.deserialize_coords(idx, self.num_cols)
                        self.put_obj(Goal(self.world, self.GOAL_COLOR_IDX), i, j)
                        pos = [i, j]

                    self.goal_locations.append(pos)
                    agent_idx += 1
                    if agent_idx >= len(self.agents):
                        break
            print("Done adding goals!")

    def _reward(self, i, rewards, reward=1):
        # for j, a in enumerate(self.agents):
        # a = self.agents[i]
        # g = self.goal_locations[i]
        # if a.pos == g.pos:
        #     rewards[i] += 20
        # if a.index == i or a.index == 0:
        rewards[i] += reward
        # if self.zero_sum:
        # if a.index != i or a.index == 0:
        #     rewards[i] -= reward

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        if done == True:
            print("Done! Summary of episode = ")
            print("Average reward = ", np.mean(rewards))
            print("Number of agents that reached goal = ", np.sum(rewards > 20))
        return obs, rewards, done, info
