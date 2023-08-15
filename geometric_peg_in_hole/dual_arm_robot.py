import collections
import time
import numpy as np
import pybullet as p
import pybullet_planning as pp
import transforms3d as t3d
import os

class DualArmRobot():
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = ori
        
        self.id = pp.load_pybullet(os.path.dirname(os.path.abspath(__file__)) + '/assets/dual_arm/dual_arm.urdf', fixed_base=True)
        pp.set_point(self.id, self.base_pos)
        pp.set_quat(self.id, self.base_ori)
        pp.dump_body(self.id)

        self.gripper_range = [0, 0.085]
        self.gripper_joint_multipliers = [1, 1, 1, 1, -1, -1]

        self.right_eef_link = pp.link_from_name(self.id, 'right_tool0')
        self.left_eef_link = pp.link_from_name(self.id, 'left_tool0')
        self.right_arm_joints = pp.joints_from_names(self.id, ['right_shoulder_pan_joint',
                                                                'right_shoulder_lift_joint',
                                                                'right_elbow_joint',
                                                                'right_wrist_1_joint',
                                                                'right_wrist_2_joint',
                                                                'right_wrist_3_joint',])
        self.left_arm_joints = pp.joints_from_names(self.id, ['left_shoulder_pan_joint',
                                                              'left_shoulder_lift_joint',
                                                              'left_elbow_joint',
                                                              'left_wrist_1_joint',
                                                              'left_wrist_2_joint',
                                                              'left_wrist_3_joint',])
        self.right_gripper_joints = pp.joints_from_names(self.id, ['right_robotiq_85_left_knuckle_joint',
                                                                   'right_robotiq_85_right_knuckle_joint',
                                                                   'right_robotiq_85_left_inner_knuckle_joint',
                                                                   'right_robotiq_85_right_inner_knuckle_joint',
                                                                   'right_robotiq_85_left_finger_tip_joint',
                                                                   'right_robotiq_85_right_finger_tip_joint',])
        self.left_gripper_joints = pp.joints_from_names(self.id, ['left_robotiq_85_left_knuckle_joint',
                                                                  'left_robotiq_85_right_knuckle_joint',
                                                                  'left_robotiq_85_left_inner_knuckle_joint',
                                                                  'left_robotiq_85_right_inner_knuckle_joint',
                                                                  'left_robotiq_85_left_finger_tip_joint',
                                                                  'left_robotiq_85_right_finger_tip_joint',])
        self.reset()

    def reset(self):
        self._step = 0
        self.vel = None
        self.obs = None
        self.hist = []
        self.i_step = 0
        self.right_arm_plan = []
        self.left_arm_plan = []
        self.right_gripper_plan = []
        self.left_gripper_plan = []
        self.step()

    def move_right_gripper(self, gripper_length):
        gripper_length = np.clip(gripper_length, *self.gripper_range)
        gripper_angle = 0.715 - np.arcsin((gripper_length - 0.010) / 0.1143)
        for joint, multiplier in zip(self.right_gripper_joints, self.gripper_joint_multipliers):
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition=gripper_angle * multiplier,
                                    force=100, maxVelocity=100)
            
    def move_left_gripper(self, gripper_length):
        gripper_length = np.clip(gripper_length, *self.gripper_range)
        gripper_angle = 0.715 - np.arcsin((gripper_length - 0.010) / 0.1143)
        for joint, multiplier in zip(self.left_gripper_joints, self.gripper_joint_multipliers):
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition=gripper_angle * multiplier,
                                    force=100, maxVelocity=100)
    
    def set_right_arm(self, arm_angles):
        for joint, angle in zip(self.right_arm_joints, arm_angles):
            p.resetJointState(self.id, joint, angle)
        self.move_right_arm(arm_angles, 100)
        
    def set_left_arm(self, arm_angles):
        for joint, angle in zip(self.left_arm_joints, arm_angles):
            p.resetJointState(self.id, joint, angle)
        self.move_left_arm(arm_angles, 100)

    def move_right_arm(self, arm_angles, max_velocity):
        for joint, angle in zip(self.right_arm_joints, arm_angles):
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition=angle, force=100, maxVelocity=max_velocity)
    
    def move_left_arm(self, arm_angles, max_velocity):
        for joint, angle in zip(self.left_arm_joints, arm_angles):
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition=angle, force=100, maxVelocity=max_velocity)

    def set_right_eef(self, pos, ori):
        ori = ori[[1, 2, 3, 0]]
        ori /= np.linalg.norm(ori)
        joint_angles = p.calculateInverseKinematics(self.id, self.right_eef_link, pos, ori, maxNumIterations=2000)
        self.set_right_arm(joint_angles[:6])
    
    def set_left_eef(self, pos, ori):
        ori = ori[[1, 2, 3, 0]]
        ori /= np.linalg.norm(ori)
        joint_angles = p.calculateInverseKinematics(self.id, self.left_eef_link, pos, ori, maxNumIterations=2000)
        self.set_left_arm(joint_angles[12:18])

    def joint_move_right_eef(self, pos, ori, max_distance=0.1, randomized=False):
        ori = ori[[1, 2, 3, 0]]
        # if randomized:
        #     pos = pos + np.random.normal(0, 0.01, 3)
        #     ori = ori + np.random.normal(0, 0.01, 4)
        ori /= np.linalg.norm(ori)
        joint_angles = p.calculateInverseKinematics(self.id, self.right_eef_link, pos, ori, maxNumIterations=2000)
        self.right_arm_plan = pp.plan_joint_motion(self.id, self.right_arm_joints, end_conf=joint_angles[:6], max_distance=max_distance)
        if self.right_arm_plan is None:
            self.right_arm_plan = []
            return False
        else:
            # self.randomized = randomized
            if randomized:
                self.right_arm_plan = [plan + np.random.normal(0, 0.01, 6)
                                       for i_plan, plan in enumerate(self.right_arm_plan)] + [self.right_arm_plan[-1]]
            pp.set_joint_positions(self.id, self.right_arm_joints, self.right_arm_plan[0][:6])
            return True
    
    def joint_move_left_eef(self, pos, ori, max_distance=0.1, randomized=False):
        ori = ori[[1, 2, 3, 0]]
        # if randomized:
        #     pos = pos + np.random.normal(0, 0.01, 3)
        #     ori = ori + np.random.normal(0, 0.01, 4)
        ori /= np.linalg.norm(ori)
        joint_angles = p.calculateInverseKinematics(self.id, self.left_eef_link, pos, ori, maxNumIterations=2000)
        self.left_arm_plan = pp.plan_joint_motion(self.id, self.left_arm_joints, end_conf=joint_angles[12:18], max_distance=max_distance)
        if self.left_arm_plan is None:
            self.left_arm_plan = []
            return False
        else:
            if randomized:
                self.left_arm_plan = [plan + np.random.normal(0, 0.01, 6)
                                       for i_plan, plan in enumerate(self.left_arm_plan)] + [self.left_arm_plan[-1]]
            pp.set_joint_positions(self.id, self.left_arm_joints, self.left_arm_plan[0][:6])
            return True
        
    def move_right_eef(self, pos, ori):
        ori = ori[[1, 2, 3, 0]]
        ori /= np.linalg.norm(ori)
        joint_angles = p.calculateInverseKinematics(self.id, self.right_eef_link, pos, ori, maxNumIterations=2000)
        self.move_right_arm(joint_angles[:6], 100)
    
    def move_left_eef(self, pos, ori):
        ori = ori[[1, 2, 3, 0]]
        ori /= np.linalg.norm(ori)
        joint_angles = p.calculateInverseKinematics(self.id, self.left_eef_link, pos, ori, maxNumIterations=2000)
        self.move_left_arm(joint_angles[12:18], 100)

    def get_obs(self):
        return self.obs
    
    def get_vel(self):
        return self.vel
        
    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()
        # print(self.joints)
    
    def update(self, max_velocity=100):
        right_eef_pos, right_eef_ori = pp.get_link_pose(self.id, self.right_eef_link)
        left_eef_pos, left_eef_ori = pp.get_link_pose(self.id, self.left_eef_link)
        right_arm_angles = [pp.get_joint_state(self.id, joint).jointPosition for joint in self.right_arm_joints]
        left_arm_angles = [pp.get_joint_state(self.id, joint).jointPosition for joint in self.left_arm_joints]
        right_gripper_length = 0.1143 * np.sin(0.715 - pp.get_joint_state(self.id, self.right_gripper_joints[0]).jointPosition) + 0.010
        left_gripper_length = 0.1143 * np.sin(0.715 - pp.get_joint_state(self.id, self.left_gripper_joints[0]).jointPosition) + 0.010

        right_eef_ori = [right_eef_ori[i] for i in [3, 0, 1, 2]]
        left_eef_ori = [left_eef_ori[i] for i in [3, 0, 1, 2]]

        self.obs = {'right_eef_pos': right_eef_pos, 'right_eef_ori': right_eef_ori,
                    'left_eef_pos': left_eef_pos, 'left_eef_ori': left_eef_ori,
                    'right_arm_angles': right_arm_angles, 'left_arm_angles': left_arm_angles,
                    'right_gripper_length': right_gripper_length, 'left_gripper_length': left_gripper_length}
        
        self.i_step += 1
        if len(self.right_arm_plan) > 0:
            self.move_right_arm(self.right_arm_plan[0], max_velocity)
            # self.set_right_arm(self.right_arm_plan[0])
            # if np.allclose(self.right_arm_plan[0], pp.get_joint_positions(self.id, self.right_arm_joints), atol=0.01):
            if self.i_step % 6 == 0:
                self.right_arm_plan.pop(0)
        if len(self.left_arm_plan) > 0:
            self.move_left_arm(self.left_arm_plan[0], max_velocity)
            # self.set_left_arm(self.left_arm_plan[0])
            # if np.allclose(self.left_arm_plan[0], pp.get_joint_positions(self.id, self.left_arm_joints), atol=0.01):
            if self.i_step % 6 == 0:
                self.left_arm_plan.pop(0)
        if len(self.right_gripper_plan) > 0:
            self.move_right_gripper(self.right_gripper_plan[0], max_velocity)
            if np.isclose(self.right_gripper_plan[0], right_gripper_length, atol=0.01):
                self.right_gripper_plan.pop(0)
        if len(self.left_gripper_plan) > 0:
            self.move_right_gripper(self.left_gripper_plan[0], max_velocity)
            if np.isclose(self.left_gripper_plan[0], left_gripper_length, atol=0.01):
                self.left_gripper_plan.pop(0)

    def step(self):
        pass