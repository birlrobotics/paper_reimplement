import Recovery_RL_Agent
import env_robot
import numpy as np
# import torch
import region


robot = env_robot.Env(dim=2)
agent = Recovery_RL_Agent.Agent(dim=2)

demo_goal =  [[10,5],[20,20],[30,40],[40,35],[45,45],[50,55]]
robot.demonstration(demo_goal)
demo_act_dict = agent.demo_record(demo_goal)

repeat_times = 2

for i in range(0,repeat_times):
    # executing the demo action and restore experience tuples in agent
    episode_record,_ = robot.execute_separate_demo_act(demo_act_dict)
    agent.exp_record(episode_record)
    # Reset env, back to start point
    robot.test_reset()


exp_tuple_test = agent.get_exp_list()
# for e in exp_tuple_test:
#     print(e.state)
experience_list = exp_tuple_test
action_dict = demo_act_dict
# print(action_dict)
# print(experience_list)

obj = region.Region_Cluster(experience_list, action_dict)
r_s = obj.learn_state_region()
print(r_s)
