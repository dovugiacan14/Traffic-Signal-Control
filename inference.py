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
    
    log_dir = f'/home/ubuntu/Desktop/TSC/logs/ingolstadt21/a3c'

    env = make_env(pad=True, begin_time=57600, num_sec=61200, log_dir=log_dir)
    register_env(f'ingolstadt21', lambda _: env)

    obs_space = env.observation_space
    act_space = env.action_space


    print("Training completed. Restoring new Trainer for action inference.")
    # Create new Trainer and restore its state from the last checkpoint.
    config={
            'env' : 'ingolstadt21',
            'multiagent' : {
                'policies' : {
                    '0' : PolicySpec(a3c_torch_policy.A3CTorchPolicy, obs_space, act_space, {})
                },
                "policy_mapping_fn" : (lambda _: '0')
            },
            # 'num_workers': 5,
            'num_gpus' : 1,
            'framework' : 'torch',
            "no_done_at_end": True
        }
    algo = A3CTrainer(config=config)
    algo.restore('/home/ubuntu/Desktop/TSC/ray_result/checkpoint_006006/checkpoint-6006')

    for i in range(100):
    # Create the env to do inference in.
        obs = env.reset()
        # print(env)
        num_episodes = 0
        episode_reward = 0.0

        done = False
        while not done:
            action = {}
            for agent_id, agent_obs in obs.items():
                policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
                action[agent_id] = algo.compute_single_action(agent_obs, policy_id=policy_id)
            obs, reward, done, info = env.step(action)
            done = done['__all__']
            # sum up reward for all agents
            episode_reward += sum(reward.values())
            action.clear()
        os.rename('result/result.csv_conn0_run1.csv', f'infer/{i}.csv')