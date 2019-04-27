from __future__ import print_function
from test_two_dim import two_dim_test

twodimtest = two_dim_test()
twodimtest.init_region_infs()

repeat_time = 3
exp_list = []
# init an object of one_dim_test
# execute the demo and get exp

for i in range(repeat_time):
    for e in twodimtest.two_dim_sample_gen():
        exp_list.append(e)
        twodimtest.reset_pos()
# Batch training
for epoch in range(100):
    for index in range(100):
        states,actions,rewards,next_states,dones = twodimtest.get_batch_sample(exp_list,batch_num = 3)
        twodimtest.q_learn(states,actions,rewards,next_states,dones,epoch)
        

print(twodimtest.get_parameters())