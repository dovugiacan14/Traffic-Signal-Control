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
from ray.rllib.agents.dqn import DQNTorchPolicy, DQNTrainer, DEFAULT_CONFIG
# from ray.rllib.agents.ppo import PPOTrainer,PPOTorchPolicy, DEFAULT_CONFIG
# from ray.rllib.agents.a3c import A3CTrainer, a3c_torch_policy, DEFAULT_CONFIG
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.policy.policy import PolicySpec
ray.init(num_gpus=1)

net = '/home/thoaican/Desktop/TSC/RESCO/ingolstadt21/ingolstadt21.net.xml'
rou = '/home/thoaican/Desktop/TSC/RESCO/ingolstadt21/ingolstadt21.rou.xml'
log_dir = '/home/thoaican/Desktop/TSC/logs/ingolstadt21/dqn_1/dqn'


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
        env = ParallelPettingZooEnv(env)
    return env

env = make_env(pad=True, begin_time=57600, num_sec=61200)
register_env('ingolstadt21_run2', lambda _: env)


if __name__ == '__main__':
    

    for seed in range(2, 10):

        log_dir = f'/home/thoaican/Desktop/TSC/logs/ingolstadt21/dqn_seed{seed}/dqn'

        env = make_env(pad=True, begin_time=57600, num_sec=61200, log_dir=log_dir)
        register_env(f'ingolstadt21_dqn_seed_{seed}', lambda _: env)

        obs_space = env.observation_space
        act_space = env.action_space

        del env
    
        tune.run(
            DQNTrainer,
            config={
                'env' : f'ingolstadt21_dqn_seed_{seed}',
                'multiagent' : {
                    'policies' : {
                        '0' : PolicySpec(DQNTorchPolicy, obs_space, act_space, {})
                    },
                    "policy_mapping_fn" : (lambda _: '0')
                },
                'seed' : seed,
                'num_workers': 20,
                'num_gpus' : 0.8,
                # 'num_gpus_per_worker' : 0.001,
                "v_min": -50.0,
                "v_max": 50.0,
                "dueling": False,
                # Dense-layer setup for each the advantage branch and the value branch
                # in a dueling architecture.
                # Whether to use double dqn
                "hiddens": [4],
                "double_q": False,
                # N-step Q learning
                "n_step": 5,
                # 'num_gpus_per_worker' : (1-0.5)//10,
                'framework' : 'torch',
                "lr": 0.001,
                "no_done_at_end": True
        },
        stop = {'timesteps_total' : 2.5e6}
        )