from gym.envs.registration import register

register(
    id="multigrid-mapf-v0",
    entry_point="gym_multigrid.envs:MapfEnv",
)
