from gym_multigrid.multigrid import *
import json
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

DIR_MAP = {
    (0, 1): 0,  # right
    (-1, 0): 1,  # up
    (0, -1): 2,  # left
    (1, 0): 3,  # down
}

class FlatMultiGridObsWrapper(gym.core.ObservationWrapper):
    """
    Use the state image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        imgSpace = env.observation_space
        self.imgSize = (
            imgSpace.shape[0]
            * imgSpace.shape[1]
            * imgSpace.shape[2]
            * imgSpace.shape[3]
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.imgSize,),
            dtype="uint8",
        )

    def observation(self, obs):
        obs_ = [img.flatten() for img in obs]
        obs_ = np.concatenate(obs_)
        # obs_ = obs_.flatten()
        return obs_


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
        agent_view_size=5,
        # map_file_path=None,
        # agent_file_path=None,
        # task_file_path=None,
        scenario_file=None,
        edge_weights=None,
        **kwargs,
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

        self.world = SmallWorld

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

        # Set edge weights. Edge weights are stored as array of shape [h, w, 4]
        # where the last dimension encodes the edge weight of the corresponding
        # node in the right, up, left, down directions.
        if edge_weights is None:
            # If no weights are provided, set all valid edge weights to 1
            edge_weights = np.ones((self.num_rows, self.num_cols, 4))
        # Set weights of all invalid edges to -1
        self.edge_weights = self._set_invalid_edge_weights(edge_weights)
        assert self.edge_weights.shape == (self.num_rows, self.num_cols, 4)

        super().__init__(
            grid_size=self.size,
            width=self.num_rows,
            height=self.num_cols,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=agent_view_size,
            objects_set=self.world,
            **kwargs,
        )

        # Incorporate edge weights in observation space
        if self.partial_obs:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    agent_view_size,
                    agent_view_size,
                    self.objects.encode_dim + 4, # plut 4 for edge weights
                    len(self.agents),
                ),
                dtype="uint8",
            )

        else:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    self.width,
                    self.height,
                    self.objects.encode_dim + 4, # plut 4 for edge weights
                    len(self.agents),
                ),
                dtype="uint8",
            )

        self.ob_dim = np.prod(self.observation_space.shape)

        # Initialize the state
        self.reset()

    def _concat_edge_weights_to_obs(self, obs):
        view_size = self.agents[0].view_size
        pad_size = view_size // 2
        padded_edge_weights = np.array([np.pad(self.edge_weights[:,:,i], pad_size, mode="constant") for i in range(4)])
        padded_edge_weights = np.moveaxis(padded_edge_weights, 0, 2)
        for i, agent in enumerate(self.agents):
            # Get edge weights in the view window, assuming partial_obs is True
            x, y = agent.pos
            slice_start_x = x
            slice_end_x = x + view_size
            slice_start_y = y
            slice_end_y = y + view_size
            partial_edge_weights = padded_edge_weights[slice_start_x:slice_end_x,slice_start_y:slice_end_y,:]
            assert partial_edge_weights.shape == (view_size, view_size, 4)

            # Append edge weights to the last 4 dim
            obs[i] = np.concatenate([obs[i], partial_edge_weights], axis=-1)
        return obs

    def reset(
        self,
        *,
        seed: int = None,
        options: Dict[str, Any] = None,
    ):
        obs, info = super().reset(seed=seed, options=options)
        obs = self._concat_edge_weights_to_obs(obs)
        return obs, info


    def _set_invalid_edge_weights(self, edge_weights):
        """Set weights of all invalid edges to -1"""
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                # Non-zero means non-traversable
                if self.loaded_map[i, j] != 0:
                    # All edges from the node are blocked
                    edge_weights[i, j] = -1

                # Check each neighbor
                # All edges to non-traversable neighbors are blocked
                for dx, dy in DIR_MAP.keys():
                    n_i = i + dx
                    n_j = j + dy
                    if 0 <= n_i < self.num_rows and 0 <= n_j < self.num_cols \
                        and self.loaded_map[n_i, n_j] != 0:
                        edge_weights[i, j, DIR_MAP[(dx, dy)]] = -1
        return edge_weights

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

        print("Loading from file = ", self.map_file_path)

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
        rewards[i] += reward

    def step(self, actions):
        obs, rewards, terminated, truncated, info = MultiGridEnv.step(self, actions)
        obs = self._concat_edge_weights_to_obs(obs)
        if terminated or truncated:
            print("Done! Summary of episode = ")
            print("Average reward = ", np.mean(rewards))
            print("Number of agents that reached goal = ", np.sum(rewards > 20))
        return obs, rewards, terminated, truncated, info
