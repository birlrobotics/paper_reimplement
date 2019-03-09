
agent = Agent()
robot = Env()

# How many times does the robot execute the human demonstration
repeat_times =

agent.demo_record()

demo_list = agent.get_demo_act_list()

for i in repeat_times:
    episode_record = robot.execute_demo_act(demo_list)
    agent.exp_record (episode_record)
