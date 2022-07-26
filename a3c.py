import os
import sys
from gym import make
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import sumo_rl
import ray
import supersuit

from ray.rllib.agents.a3c import A3CTrainer, a3c_torch_policy, DEFAULT_CONFIG
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.registry import get_trainer_class

ray.init(num_gpus=1)

net = '/home/ubuntu/Desktop/TSC/RESCO/ingolstadt21/ingolstadt21.net.xml'
rou = '/home/ubuntu/Desktop/TSC/RESCO/ingolstadt21/ingolstadt21.rou.xml'
# log_dir = '/home/ubuntu/Desktop/TSC/logs/ingolstadt21/a3c_1/a3c'


def make_env(single=False, pad=False, begin_time=0, num_sec=3600, log_dir=''):
    
    if not single:
        env = sumo_rl.parallel_env(net_file=net,
                                    route_file=rou,
                                    out_csv_name=log_dir,
                                    single_agent=single,
                                    use_gui=False,
                                    begin_time = begin_time,
                                    num_seconds=num_sec)
    else:
        env = sumo_rl.SumoEnvironment(net_file=net,
                                    route_file=rou,
                                    single_agent=single,
                                    out_csv_name=log_dir,
                                    use_gui=False,
                                    begin_time = begin_time,
                                    num_seconds=1000)
    if pad:
        env = supersuit.pad_observations_v0(env)
        env = supersuit.pad_action_space_v0(env)
    if not single:
        env = ParallelPettingZooEnv(env)
    return env


if __name__ == '__main__':
    
    for seed in range(0,1):

        log_dir = f'/home/ubuntu/Desktop/TSC/logs/ingolstadt21/a3c_seed_{seed}/a3c'

        env = make_env(pad=True, begin_time=57600, num_sec=61200, log_dir=log_dir)
        register_env(f'ingolstadt21_a3c_seed_{seed}', lambda _: env)

        obs_space = env.observation_space
        act_space = env.action_space

        results = tune.run(
                A3CTrainer,
                config={
                    'env' : f'ingolstadt21_a3c_seed_{seed}',
                    'multiagent' : {
                        'policies' : {
                            '0' : PolicySpec(a3c_torch_policy.A3CTorchPolicy, obs_space, act_space, {})
                        },
                        "policy_mapping_fn" : (lambda _: '0')
                    },
                    'seed' : seed,
                    'num_workers': 5,
                    'num_gpus' : 1,
                    'framework' : 'torch',
                    "lr": 0.001,
                    "no_done_at_end": True
            },
            stop = {'timesteps_total' : 8e6},
            checkpoint_freq = 10,
            checkpoint_at_end=True,
            local_dir = '/home/ubuntu/Desktop/TSC/ray_result'
            )