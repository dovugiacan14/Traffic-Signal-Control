import sumo_rl
import gym
from stable_baselines3.dqn.dqn import DQN
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_rl import SumoEnvironment
import traci


if __name__ == '__main__':

    env = SumoEnvironment(net_file='cologne8/cologne8.net.xml',
                                    route_file='cologne8/cologne8.rou.xml',
                                    out_csv_name='Town05.rou.xml',
                                    single_agent=True,
                                    use_gui=True,
                                    num_seconds=100000,
                                    max_depart_delay=0)

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.01,
        learning_starts=0,
        train_freq=1,
        target_update_interval=100,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        # verbose=1
    )
    model.learn(total_timesteps=100000)