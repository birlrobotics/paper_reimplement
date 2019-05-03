
import numpy as np
from collections import namedtuple
import region
from collections import OrderedDict


demo_act_dict = OrderedDict()
goal_tuples = np.array([-1,-1,-1])
demo_act_dict[3]=goal_tuples
demo_act_dict[4]=goal_tuples
demo_act_dict[5]=goal_tuples
demo_act_dict[7]=goal_tuples
demo_act_dict[8]=goal_tuples
demo_act_dict[9]=goal_tuples


exp_tuple_test = np.load('experience_tuple_no_recovery_skill_positions.npy', encoding="latin1")


experience_tuple = []
experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
    
for i in range(len(exp_tuple_test)):
    action = OrderedDict()

    state = exp_tuple_test[i][0]
    action[exp_tuple_test[i][1]] = goal_tuples
    reward = exp_tuple_test[i][2]
    next_state = exp_tuple_test[i][3]
    done = exp_tuple_test[i][4]
    e =  experience(state, action, reward, next_state, done)
    experience_tuple.append(e)

experience_list = experience_tuple
action_dict = demo_act_dict

obj = region.Region_Cluster()
clustered_regions = obj.cluster_for_visualize(experience_list, action_dict)

