
agent = Agent()
robot = Env()
# How many times does the robot execute the human demonstration
repeat_times = 100

#  record demo and return a demonstration action list
agent.demo_record()
demo_act_list = agent.get_demo_act_list()

for i in repeat_times:
    # executing the demo action and restore experience tuples in agent
    episode_record = robot.execute_demo_act(demo_act_list)
    agent.exp_record(episode_record)

exp_tuple_list = agent.get_exp_list()

#  Get a set of regions. Each region is a set of a exp tuples
region_cluster = Region_Cluster(exp_tuple_list, demo_act_list)

# Use Gaussian maximum likelihood  for estimating the region distribution probability
# Get 
agent.gaussian_likelihood(region_cluster)
