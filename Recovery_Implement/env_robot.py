from math import sqrt
import numpy as np

class Env():

    def __init__(self,dim):
        self.goals_list= []
        #  Init the beginnig position of robot.
        self.current_pos = np.zeros(dim)
        self.dim = dim
        #  test_perturbation_count used for generate virtual contact mode.
        self.test_perturbation_count = 0
        self.final_goal_state = 0
        # self.test_final_goal_state = np.array((300,300))
        self.have_final_goal = False

        self.original_skills_amount= 0
#   Reset the environment to initial state.

    def demonstration(self,demo_states_list):
        """Record the demonstration state ponit into a goal list.
        Param:
            demo_states_list: input a list of the demonstration position tuples.
        ===========================
        Return:
            self.goals_list(list): the list of the goals of demonstration actions.
        """
        self.goals_list = demo_states_list
        self.final_goal_state = demo_states_list[-1]
        self.original_skills_amount = len(demo_states_list)
        print("Env: the original demonstration skills amount is:{} \n".format( self.original_skills_amount))

        print("Env: The demonstration is:")
        for i,goal in enumerate(self.goals_list):
            print("point {}: {}".format(i,goal))
        print("\n")
        print("Env: The final goal position is:{}".format(self.final_goal_state))
        return self.goals_list

    def get_goals_list(self):
        """Return the demonstration goal list to agent.
        """
        return self.goals_list

    def reset_goals_list(self):
        """Reset the demonstration goal list
        """
        self.goals_list= []
        self.have_final_goal = False

    def append_new_final_goal(self):
        """Append a new final goal.  
        """
        self.goals_list.append(ROS_current_pos())
        self.have_final_goal = True
        print("Append a new fianl goal:",ROS_current_pos())

    def test_reset(self):
        """Reset the robot to original point.
        """
        self.current_pos = [0]*self.dim

    def state_distance(self,p1,p2):
        """Calculate the distance of two input points.
        Param:
            s1(2D or 3D array): the position of ponit 1
            s2(2D or 3D array): the position of ponit 1
        ===========================
        Return:
            reward(float)
        ===========================
        Note: The reward estimation includes the duration of the relative action execution, 
              and whether the next state is within the goal region G.  
        """
        distance = 0
        for i,j in zip(p1,p2):
            distance += abs(i - j)**2
        distance = sqrt(distance)
        return distance

    def get_reward(self, position, next_position, duration, task_type = 'maze'):
        """Get the reward and dones.
        Param:
            position(2D or 3D array): the position of state
            next_position(2D or 3D array): the position of next state
            duration(float): the action execution duration
            task_type(string): the type of the task
        ===========================
        Return:
            reward(float)
            done(Boolen): indication of whether the next state is terminal state.
        ===========================
        Note: The reward estimation includes the duration of the relative action execution, 
              and whether the next state position is within the goal region G.  
        """
        distance = self.state_distance(position,next_position)
        r= distance
        
        # if the next state position is in goal region G, which is set in a circle with radius 20.
        if task_type == 'maze':
            if self.state_distance(self.final_goal_state, next_position) > 20:
                r += duration
            else:
                return -r, True
        return -r, False

    def perturbation(self, delta_d):
        """Executing the reversible perturbation with parameter delta.
        Param:
            delta_d(float): the parameter of perturbation
        ===============================
        Return:
            contact_mode(4D or 6D array): the contact mode
        ===============================
        Note: The reward estimation includes the duration of the relative action execution, 
              and whether the next state is within the goal region G.  
        """
        contact_mode = []

        current_s = ROS_current_pos()

        desired_1 = current_s + delta_d
        desired_2 = current_s - delta_d
        desired_list = [desired_1,desired_2]

        for desired_s in desired_list:

            ROS_move_to(desired_s)
            result_s = ROS_current_pos() - current_s
            contact_s = result_s/delta_d
            contact_mode.append(contact_s)
            ROS_move_to(current_s)

        return contact_mode

    def test_pertubation(self, add_pert = False, delta = 10):
        """Executing the reversible perturbation with parameter delta.(for test)
        Param:
            delta(float): the parameter of perturbation
        ===============================
        Return:
            per(4D or 6D array): the contact mode
        ===========================
        Note: The reward estimation includes the duration of the relative action execution, 
              and whether the next state is within the goal region G.  
        """
        if add_pert:
            a= self.test_perturbation_count
            # The begining position is a constant.(Because the robot is always reset to a certain position)
            if a == 0:
                pert = np.zeros( self.dim *2)
            else:
                pert = np.random.normal(0.25 * a, 0.07, self.dim *2)

            self.test_perturbation_count += 1
            if self.test_perturbation_count == (self.original_skills_amount+1):
                self.test_perturbation_count = 0

        else: pert = np.zeros( self.dim *2)
        
        return pert

    def test_duration_func(self,distance,ee_speed = 5):
        """Return the duration of a specific execution of an action.
        Param:
            distance: the execution moving distance
            ee_speed: the speed of end_effector
        ===============================
        Return:
            duration time = distance/ee_speed
        ===========================
        Note: It just for test, and the real duration should be get from ROS.
        """
            # assume the end effector speed is
        return distance/ee_speed

    def test_robot_move(self, goal,  mean = 0, std = 1, add_noise = True):
        """Move the robot end effector to specific postion, and return the time of duration.

        Param:
        goal(2D or 3D array): the goal positon end effector to move to
        ===============================
        Return:
        duration: the action execution duration time (second).
        """
        # # Make two noise region on goal (100,300)
        # multi_region = np.array((100,300))
        # if (goal == multi_region).all():
        #     if np.random.randn((1)) >= 1 :
        #         mean = 10
        #     elif np.random.randn((1)) <= -1:
        #         mean = -10
        #     elif 0>= np.random.randn((1)) > -1:
        #         mean = 5
        #     else :
        #         mean = -5

        if add_noise:
            noise_goal = goal + np.random.normal(mean, std, self.dim)
        else:
            noise_goal = goal

        distance = self.state_distance(noise_goal, self.current_pos)
        duration = self.test_duration_func(distance)
        self.current_pos = noise_goal
        return duration


    def robot_move(self, goal):
        """Move the robot end effector to specific postion, and return the time of duration.
        Param:
            goal(2D or 3D array): the goal positon end effector to move to
        ===============================
        Return:
            duration: the action execution duration time (second).
        """
        ROS_move_to(goal)
        self.current_pos = ROS_current_pos()
        duration = ROS_Execution_Duration()
        return duration

    def test_get_pos(self):
        """Return the current position of end effector. (for test)
        """
        return self.current_pos

    def get_pos(self):
        """Return the current position of end effector. (for test)
        """
        self.current_pos = ROS_current_pos()
        return self.current_pos



# robot executes the demo action
# arg:
#       execute_demo_act_list: input a list of action for execution.
# return:
#       episode_record: An episode experience. Restored as a list of
#       namedtuple [(s,z,a,r,s',z',),(s,z,a,r,s',z',),(s,z,a,r,s',z',),...]

    def execute_seperate_demo_act(self,execute_demo_act_dict):
        """Robot executes the demo action
        Param:
            execute_demo_act_list(list): input a list of action for execution.
        ===============================
        Return:
            episode_record(list): An episode experience. 
        ================================
        Note: The episode experience is a list of namedtuple [(state,action,reward,next_state),...]
                And the state includes the position and contact mode.
        """
        episode_record= []
        seperate_s_c_episode_record = []
        # cache_exp_tuple = ()

        position = self.current_pos
        contact = self.test_pertubation(add_pert=True)
        state = np.append(position,contact)

        # state = self.current_pos

        for  act_index, act_goal in execute_demo_act_dict.items():

#           Noise move means adding the Gaussian noise to the goal position of an action,
#           to model the mechanical or control error.
            # init an action dict with relative value
            action={}
            action[act_index] = act_goal
            # executing the action and return the duration (duration should be get from ROS)
            exe_duration = self.test_robot_move(act_goal)

            next_position = self.current_pos
            next_contact = self.test_pertubation(add_pert=True)
            next_state = np.append(next_position,next_contact)
            # next_state =self.current_pos

            reward, done = self.get_reward(position, next_position, exe_duration)
            # reward, done = self.get_reward(state, next_state, exe_duration)

            exp_tuple = (state, action, reward, next_state, done)
            seperate_s_c_tuple = (position,contact, action, reward, next_position,next_contact, done)

            episode_record.append(exp_tuple)
            seperate_s_c_episode_record.append(seperate_s_c_tuple)

            position = next_position
            contact = next_contact
            state = next_state

        return episode_record, seperate_s_c_episode_record




#     def execute_demo_act(self,execute_demo_act_dict):
#         """Robot executes the demo action
#         Param:
#             execute_demo_act_list(list): input a list of action for execution.
#         ===============================
#         Return:
#             episode_record(list): An episode experience. 
#         ================================
#         Note: The episode experience.is a list of namedtuple [(state,action,reward,next_state),...]
#                 And the state includes the position and contact mode.
#         """
#         episode_record= []

#         state = self.current_pos

#         for  act_index, act_goal in execute_demo_act_dict.items():

# #           Noise move means adding the Gaussian noise to the goal position of an action,
# #           to model the mechanical or control error.
#             # init an action dict with relative value
#             action={}
#             action[act_index] = act_goal
#             # executing the action and return the duration (duration should be get from ROS)
#             exe_duration = self.test_robot_move(act_goal)

#             next_state = self.current_pos

#             reward, done = self.get_reward(state, next_state, exe_duration)

#             exp_tuple = (state, action, reward, next_state, done)
#             episode_record.append(exp_tuple)

#             state = next_state

#         return episode_record