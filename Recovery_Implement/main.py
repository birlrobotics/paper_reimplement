"""
"""
agent = Agent()
robot = Env()
# How many times does the robot execute the human demonstration
repeat_times = 100

#  record demo and return a demonstration action list
agent.demo_record()
demo_act_dict = agent.get_demo_act_list()

for i in repeat_times:
    # executing the demo action and restore experience tuples in agent
    episode_record = robot.execute_demo_act(demo_act_dict)
    agent.exp_record(episode_record)
    # Reset env
    robot.reset()

#  Clustering the experience tuples
agent.learn_cluster_region()


# Use Gaussian maximum likelihood  for estimating the region distribution probability
# Return a list of namedptuple [(region_index,mean,std,is_goal)...]
agent.gaussian_likelihood()

agent.init_value_function()

#  Learn initial policy
agent.learn_initial_policy()
