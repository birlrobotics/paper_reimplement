from __future__ import print_function
from test_one_dim import one_dim_test


repeat_time = 3000
exp_list = []
# init an object of one_dim_test
testc = one_dim_test()
# execute the demo and get exp
for i in range(repeat_time):
    for e in testc.one_dim_sample_gen():
        exp_list.append(e)
        testc.reset_pos()

# Batch training
for epoch in range(10):
    for index in range(100):
        states,actions,rewards,next_states,dones = testc.get_batch_sample(exp_list,batch_num = 3)
        testc.q_learn(states,actions,rewards,next_states,dones,epoch)
        print

print(testc.get_parameters())