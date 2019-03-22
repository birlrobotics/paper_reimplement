import Recovery_RL_Agent
import env_robot
import numpy as np

robot = env_robot.Env(dim=2)
agent = Recovery_RL_Agent.Agent()

goal_array = np.array(([100,100],[100,300],[300,300]))
demo_act_dict = agent.demo_record(goal_array)

repeat_times = 1000

for i in range(0,repeat_times):
    # executing the demo action and restore experience tuples in agent
    episode_record = robot.execute_demo_act(demo_act_dict)
    agent.exp_record(episode_record)
    # Reset env, back to start point
    robot.test_reset()


covar = np.array([[1,0],[0,1]])
phi_inf_list = []
for a_dict,mean in zip(demo_act_dict.items(),goal_array):
    action_dict={}
    action_dict[a_dict[0]] = a_dict[1]
    phi_i = (0,action_dict,False,mean,covar)
    phi_inf_list.append(phi_i)


agent.test_init_value_function(phi_inf_list)

agent.test_learn_initial_policy()