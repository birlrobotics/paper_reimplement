import Recovery_RL_Agent
import env_robot
import numpy as np
# import torch
import region


robot = env_robot.Env(dim=2)
agent = Recovery_RL_Agent.Agent()


goal_array = np.array(([100,300],[100,500],[300,500],[300,300],[300,200],[300,100]))
demo_act_dict = agent.demo_record(goal_array)

repeat_times = 100

for i in range(0,repeat_times):
    # executing the demo action and restore experience tuples in agent
    episode_record = robot.execute_demo_act(demo_act_dict)
    agent.exp_record(episode_record)
    # Reset env, back to start point
    robot.test_reset()


exp_tuple_test = agent.get_exp_list()

experience_list = exp_tuple_test
action_dict = demo_act_dict
print(action_dict)
obj = region.Region_Cluster(experience_list, action_dict)
r_s = obj.learn_state_region()
print(r_s)
