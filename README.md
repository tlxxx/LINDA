# LINDA in PyMARL framework

This is the code of [LINDA: Multi-Agent Local Information
Decomposition for Awareness of Teammates](https://arxiv.org/pdf/2109.12508) in PyMARL framwork.

## Run an experiment 

To run the experiments, you can just input the command in your terminal:

```shell
python3 src/main.py --config=your_alg --env-config=sc2 with env_args.map_name=8m
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

The project retained all the features of the PyMARL project and provide image that describe the relationship between loss, battle_won_rate, test_battle_won_rate and t_env in results/plog/your_algo.

## Some claims

This project only implemented LINDA_QMIX and LINDA_VDN additionally and please contact me if there is bugs in the project.

QWQ
# LINDA
