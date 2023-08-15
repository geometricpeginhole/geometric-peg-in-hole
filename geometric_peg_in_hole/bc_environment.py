import pybullet as p
import pybullet_planning as pp
import geometric_peg_in_hole.dual_arm_robot
import geometric_peg_in_hole.realsense_camera
import os
import numpy as np
import transforms3d as t3d
import matplotlib.colors
import glob
import pkgutil
# import geometric_peg_in_hole.pybullet_recorder


class BCEnvironment():
    def __init__(self, args, variant, object_set, suffix='_preprocessed.urdf', traj_type=1) -> None:
        sim_id = pp.connect(use_gui=True, width=1280, height=720)
        # sim_id = pp.connect(use_gui=False, width=1280, height=720)
        # egl = pkgutil.get_loader('eglRenderer')
        # plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        # print("plugin=", plugin)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, True, physicsClientId=sim_id)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, True, physicsClientId=sim_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, True, physicsClientId=sim_id)
        pp.set_camera(args.camera_yaw, args.camera_pitch, args.camera_distance, args.camera_target_position)

        self.x_pos = args.x_pos
        self.x_align = args.x_align
        self.x_close = args.x_close
        self.y_pos = args.y_pos
        self.z_pos = args.z_pos
        self.x_align_step = args.x_align_step
        self.n_wait_threshold = args.n_wait_threshold
        # self.recorder = geometric_peg_in_hole.pybullet_recorder.PybulletRecorder()

        self.obj_random_y_pos = variant.obj_random_y_pos
        self.obj_random_z_pos = variant.obj_random_z_pos
        self.obj_random_y_rot = variant.obj_random_y_rot
        self.obj_random_z_rot = variant.obj_random_z_rot

        if args.obj_color is not None:
            self.obj_color = matplotlib.colors.to_rgba(args.obj_color)
            self.base_ids_ext_id = None
        else:
            self.base_ids_ext_id = {}
            self.obj_color = None
        self.max_velocity = args.max_velocity
        self.randomized = args.randomized
        self.traj_type = traj_type

        self.object_set = object_set

        self.load(args.model_dir, suffix)
        self.reset(0)

    def load(self, model_dir, suffix):
        self.robot = geometric_peg_in_hole.dual_arm_robot.DualArmRobot([0, 0, 0], pp.quat_from_euler([-np.pi / 2, np.pi / 2, 0]))
        self.wood_texture = p.loadTexture(os.path.dirname(os.path.abspath(__file__)) + '/assets/dual_arm/light-wood.png')
        p.changeVisualShape(0, 43, textureUniqueId=self.wood_texture)

        self.camera_top = geometric_peg_in_hole.realsense_camera.RealsenseCamera()
        self.camera_left = geometric_peg_in_hole.realsense_camera.RealsenseCamera()
        self.camera_right = geometric_peg_in_hole.realsense_camera.RealsenseCamera()
        self.eef_T_camera = t3d.affines.compose(np.array([0, 0.15, 0]), t3d.euler.euler2mat(np.pi * 7 / 8, 0, np.pi), [1, 1, 1])

        urdf_dir = os.path.dirname(os.path.abspath(__file__)) + '/' + model_dir
        self.target_ids = []
        if self.obj_color is None:
            self.base_ids_ext_id
            target_obj_paths = glob.glob(urdf_dir + '/*_base_preprocessed.obj')
            target_obj_paths = sorted(target_obj_paths, key=lambda x: ('Z' if 'rotated' in x else 'A') + x)
            for target_obj_path in target_obj_paths:
                base_visual = p.createVisualShape(p.GEOM_MESH,
                                                fileName=target_obj_path,
                                                rgbaColor=(1, 1, 1, 1),
                )
                if 'cap' in os.path.basename(target_obj_path):
                    ext_visual = p.createVisualShape(p.GEOM_MESH,
                                                    fileName=target_obj_path.replace('_base_preprocessed.obj', '_ext_preprocessed.obj'),
                                                    rgbaColor=(0.5, 0.5, 0.5, 1),
                    )
                else:
                    ext_visual = p.createVisualShape(p.GEOM_MESH,
                                                    fileName=target_obj_path.replace('_base_preprocessed.obj', '_ext_preprocessed.obj'),
                                                    rgbaColor=(0.0, 0.0, 0.0, 1),
                    )
                base_id = p.createMultiBody(baseMass=0.15,
                                            baseVisualShapeIndex=base_visual,)
                ext_id = p.createMultiBody(baseMass=0.15,
                                            baseVisualShapeIndex=ext_visual,
                )
                self.target_ids.append(base_id)
                self.base_ids_ext_id[base_id] = ext_id
        else:
            target_obj_paths = glob.glob(urdf_dir + '/*_preprocessed.obj')
            target_obj_paths = sorted(target_obj_paths, key=lambda x: ('Z' if 'rotated' in x else 'A') + x)
            for target_obj_path in target_obj_paths:
                obj_visual = p.createVisualShape(p.GEOM_MESH,
                                                fileName=target_obj_path,
                                                rgbaColor=self.obj_color,
                )
                object_id = p.createMultiBody(baseMass=0.15,
                                            #   baseCollisionShapeIndex=obj_collision,
                                            baseVisualShapeIndex=obj_visual,)
                self.target_ids.append(object_id)
            # self.recorder.register_object(object_id, target_urdf_path)
        # arrow circle cross diamond hexagon key line pentagon U
        self.rot_indices = [[0, 1, 2, 3], [0], [0], [0, 1], [0, 1], [0, 1, 2, 3], [0, 1], [0, 1, 2, 3], [0, 1, 2, 3]] * 2


    def reset(self, seed):
        rng = np.random.Generator(np.random.PCG64(seed))

        for target in self.target_ids:
            pp.set_pose(target, (np.array([5, 5, 0]), pp.quat_from_euler([0, 0, 0])))
            pp.set_velocity(target, (np.array([0, 0, 0]), pp.quat_from_euler([0, 0, 0])))
            if self.base_ids_ext_id is not None:
                ext_target = self.base_ids_ext_id[target]
                pp.set_pose(ext_target, (np.array([5, 5, 0]), pp.quat_from_euler([0, 0, 0])))
                pp.set_velocity(ext_target, (np.array([0, 0, 0]), pp.quat_from_euler([0, 0, 0])))

        self.robot.reset()
        self.robot.move_left_gripper(0.075)
        self.robot.move_right_gripper(0.075)
        self.robot.set_right_arm(np.deg2rad([86, -54, -137, 7, 150, -8]))
        self.robot.set_left_arm(np.deg2rad([256, -142, 124, -142, -125, 183]))

        self.robot.update()
        robot_obs = self.robot.get_obs()
        self.world_T_eef_init = [pose_to_T(robot_obs['left_eef_pos'], robot_obs['left_eef_ori']),
                                 pose_to_T(robot_obs['right_eef_pos'], robot_obs['right_eef_ori'])]
        
        curr_target_indices = rng.choice(self.object_set)
        curr_target_indices = [curr_target_indices * 2, curr_target_indices * 2 + 1]
        if rng.uniform(0, 1) < 0.5:
            curr_target_indices = [curr_target_indices[1], curr_target_indices[0]]
        self.curr_target_indices = curr_target_indices
        
        y_pos = rng.uniform(-0.01, 0.01, 2) if self.obj_random_y_pos else [0, 0]
        z_pos = rng.uniform(0.15, 0.17, 2) if self.obj_random_z_pos else [0.15, 0.15]
        y_rot = rng.uniform(-np.pi / 16, np.pi / 16, 2) if self.obj_random_y_rot else [0, 0]
        z_rot_idx = rng.choice(self.rot_indices[self.curr_target_indices[0] // 2], 2) if self.obj_random_z_rot else [0, 0]
        eef_t_target = [np.array([0, y_pos[i], z_pos[i]]) for i in range(2)]
        eef_R_target = [t3d.euler.euler2mat(y_rot[i], 0, (z_rot_idx[i] - 1) * np.pi / 2)
                        @ t3d.euler.euler2mat(np.pi / 2, 0, 0) if z_rot_idx[i] % 2 == 1 else
                        t3d.euler.euler2mat(0, y_rot[i], (z_rot_idx[i] - 1) * np.pi / 2)
                        @ t3d.euler.euler2mat(np.pi / 2, 0, 0) for i in range(2)]
        self.eef_T_target = [t3d.affines.compose(eef_t_target[i], eef_R_target[i], [1, 1, 1]) for i in range(2)]
        target_T_eef = [np.linalg.inv(self.eef_T_target[0]), np.linalg.inv(self.eef_T_target[1])]

        world_T_target_align = [t3d.affines.compose([self.x_pos - self.x_align + rng.uniform(-0.01, 0.01),
                                                     self.y_pos + rng.uniform(-0.01, 0.01),
                                                     self.z_pos + rng.uniform(-0.01, 0.01)],
                                                     t3d.euler.euler2mat(0, 0, -np.pi / 2), [1, 1, 1]),
                                t3d.affines.compose([self.x_pos + self.x_align + rng.uniform(-0.01, 0.01),
                                                     self.y_pos + rng.uniform(-0.01, 0.01),
                                                     self.z_pos + rng.uniform(-0.01, 0.01)],
                                                     t3d.euler.euler2mat(0, 0, np.pi / 2), [1, 1, 1])]
        
        if self.traj_type == 1:
            align_target_T_eef = target_T_eef
        else:
            eef_t_target = [np.array([0, 0, 0.15]) for i in range(2)]
            eef_R_target = [t3d.euler.euler2mat(0, 0, (z_rot_idx[i] - 1) * np.pi / 2)
                            @ t3d.euler.euler2mat(np.pi / 2, 0, 0) if z_rot_idx[i] % 2 == 1 else
                            t3d.euler.euler2mat(0, 0, (z_rot_idx[i] - 1) * np.pi / 2)
                            @ t3d.euler.euler2mat(np.pi / 2, 0, 0) for i in range(2)]
            align_target_T_eef = [np.linalg.inv(t3d.affines.compose(eef_t_target[i], eef_R_target[i], [1, 1, 1])) for i in range(2)]
        self.world_T_eef_align = [world_T_target_align[0] @ align_target_T_eef[0], world_T_target_align[1] @ align_target_T_eef[1]]

        world_T_target_close = [t3d.affines.compose([self.x_pos - self.x_close, self.y_pos, self.z_pos], t3d.euler.euler2mat(0, 0, -np.pi / 2), [1, 1, 1]),
                                t3d.affines.compose([self.x_pos + self.x_close, self.y_pos, self.z_pos], t3d.euler.euler2mat(0, 0, np.pi / 2), [1, 1, 1])]
        self.world_T_eef_close = [world_T_target_close[0] @ target_T_eef[0], world_T_target_close[1] @ target_T_eef[1]]

        # self.stage = 'init'
        self.stage = 'rotate'
        self.done = False
        self.success = False
        self.i_step = 0

        world_T_target = [self.world_T_eef_init[0] @ self.eef_T_target[0], self.world_T_eef_init[1] @ self.eef_T_target[1]]
        for i_obj in range(2):
            pp.set_pose(self.target_ids[self.curr_target_indices[i_obj]], (world_T_target[i_obj][:3, 3], t3d.quaternions.mat2quat(world_T_target[i_obj][:3, :3])))
            if self.base_ids_ext_id is not None:
                ext_body = self.base_ids_ext_id[self.target_ids[self.curr_target_indices[i_obj]]]
                pp.set_pose(ext_body, (world_T_target[i_obj][:3, 3], t3d.quaternions.mat2quat(world_T_target[i_obj][:3, :3])))

        eef_t_target_rotate = [np.array([0, 0, 0.15]), np.array([0, 0, 0.15])]
        eef_R_target_rotate = [t3d.euler.euler2mat(0, 0, -np.pi / 2) @ t3d.euler.euler2mat(np.pi / 2, 0, 0),
                                t3d.euler.euler2mat(0, 0, -np.pi / 2) @ t3d.euler.euler2mat(np.pi / 2, 0, 0)]
        eef_T_target_rotate = [t3d.affines.compose(eef_t_target_rotate[i], eef_R_target_rotate[i], [1, 1, 1]) for i in range(2)]
        world_T_target_rotate = [self.world_T_eef_init[0] @ eef_T_target_rotate[0], self.world_T_eef_init[1] @ eef_T_target_rotate[1]]
        self.world_T_eef_rotate = [world_T_target_rotate[0] @ target_T_eef[0], world_T_target_rotate[1] @ target_T_eef[1]]

        self.world_T_waypoints = [self.world_T_eef_init, self.world_T_eef_align, self.world_T_eef_close]
        self.world_T_subwaypoints = []
        n_steps = [2, 2, 2]
        for i in range(len(self.world_T_waypoints) - 1):
            curr_T_waypoint = [np.linalg.inv(self.world_T_waypoints[i][k]) @ self.world_T_waypoints[i + 1][k] for k in range(2)]
            curr_t_waypoint = [curr_T_waypoint[0][:3, 3], curr_T_waypoint[1][:3, 3]]
            curr_R_waypoint = [curr_T_waypoint[0][:3, :3], curr_T_waypoint[1][:3, :3]]
            curr_axangle_waypoint = [t3d.axangles.mat2axangle(curr_R_waypoint[0]), t3d.axangles.mat2axangle(curr_R_waypoint[1])]
            for j in range(1, n_steps[i] + 1):
                curr_t_subwaypoint = [curr_t_waypoint[0] / n_steps[i] * j, curr_t_waypoint[1] / n_steps[i] * j]
                curr_R_subwaypoint = [t3d.axangles.axangle2mat(curr_axangle_waypoint[0][0], curr_axangle_waypoint[0][1] / n_steps[i] * j),
                                            t3d.axangles.axangle2mat(curr_axangle_waypoint[1][0], curr_axangle_waypoint[1][1] / n_steps[i] * j)]
                curr_T_subwaypoint = [t3d.affines.compose(curr_t_subwaypoint[0], curr_R_subwaypoint[0], [1, 1, 1]),
                                      t3d.affines.compose(curr_t_subwaypoint[1], curr_R_subwaypoint[1], [1, 1, 1])]
                world_T_subwaypoint = [self.world_T_waypoints[i][0] @ curr_T_subwaypoint[0], self.world_T_waypoints[i][1] @ curr_T_subwaypoint[1]]
                self.world_T_subwaypoints.append(world_T_subwaypoint)
        # print('world_T_subwaypoints', self.world_T_subwaypoints)

    
    def update(self, plan):
        # if self.stage == 'close':
        #     self.robot.update(0.5)
        # else:
        self.i_step += 1
        self.robot.update(self.max_velocity)
        robot_obs = self.robot.get_obs()
        world_T_eef = [t3d.affines.compose(robot_obs['left_eef_pos'], t3d.quaternions.quat2mat(np.array(robot_obs['left_eef_ori'])), [1, 1, 1]),
                    t3d.affines.compose(robot_obs['right_eef_pos'], t3d.quaternions.quat2mat(np.array(robot_obs['right_eef_ori'])), [1, 1, 1])]
        world_T_target = [world_T_eef[0] @ self.eef_T_target[0], world_T_eef[1] @ self.eef_T_target[1]]
        for i_obj in range(2):
            body = self.target_ids[self.curr_target_indices[i_obj]]
            pos = world_T_target[i_obj][:3, 3]
            ori = t3d.quaternions.mat2quat(world_T_target[i_obj][:3, :3])[[1, 2, 3, 0]]
            pp.set_pose(body, (pos, ori))
            if self.base_ids_ext_id is not None:
                ext_body = self.base_ids_ext_id[body]
                pp.set_pose(ext_body, (pos, ori))

        self.camera_left.set_pose(world_T_eef[0] @ self.eef_T_camera)
        self.camera_right.set_pose(world_T_eef[1] @ t3d.affines.compose([0, 0, 0], t3d.euler.euler2mat(0, 0, np.pi), [1, 1, 1]) @ self.eef_T_camera)

        if self.success:
            if self.stage != 'close':
                self.stage = 'close'
                self.n_wait = 0
        else:
            # self.success = np.allclose(world_T_eef[0], self.world_T_eef_close[0], atol=1e-2) and np.allclose(world_T_eef[1], self.world_T_eef_close[1], atol=1e-2)
            left_T_right_eef = np.linalg.inv(world_T_eef[0]) @ world_T_eef[1]
            left_T_right_close = np.linalg.inv(self.world_T_eef_close[0]) @ self.world_T_eef_close[1]
            dists = np.array([np.linalg.norm(world_T_eef[i][:3, 3] - self.world_T_eef_close[i][:3, 3]) for i in range(2)])
            angles = np.array([t3d.axangles.mat2axangle(np.linalg.inv(world_T_eef[i][:3, :3]) @ self.world_T_eef_close[i][:3, :3])[1] for i in range(2)])
            self.success = np.sum(np.abs(dists) < 0.01) + np.sum(np.abs(angles) < 5 / 180 * np.pi) == 4
            dist = np.linalg.norm(left_T_right_eef[:3, 3] - left_T_right_close[:3, 3])
            angle = t3d.axangles.mat2axangle(np.linalg.inv(left_T_right_eef[:3, :3]) @ left_T_right_close[:3, :3])[1]
            angle = np.abs(np.rad2deg(angle))
            # print('dist', dist, 'angle', angle)
            self.success = dist < 0.01 and angle < 5
            if self.success:
                print('dist', dist, 'angle', angle)

        if plan:
            # if len(self.world_T_subwaypoints):
            #     left_eef_pos = self.world_T_subwaypoints[0][0][:3, 3]
            #     left_eef_ori = t3d.quaternions.mat2quat(self.world_T_subwaypoints[0][0][:3, :3])
            #     right_eef_pos = self.world_T_subwaypoints[0][1][:3, 3]
            #     right_eef_ori = t3d.quaternions.mat2quat(self.world_T_subwaypoints[0][1][:3, :3])
            #     self.robot.move_left_eef(left_eef_pos, left_eef_ori)
            #     self.robot.move_right_eef(right_eef_pos, right_eef_ori)
            #     # if np.linalg.norm(left_eef_pos - robot_obs['left_eef_pos']) < 1e-3\
            #     #         and np.linalg.norm(right_eef_pos - robot_obs['right_eef_pos']) < 1e-3:
            #     if np.linalg.norm(left_eef_pos - robot_obs['left_eef_pos']) < 1e-2 and np.linalg.norm(right_eef_pos - robot_obs['right_eef_pos']) < 1e-2:
            #         self.world_T_subwaypoints.pop(0)
            # else:
            #     self.stage = 'close'
            #     self.n_wait = 0

            # if self.stage == 'close':
            #     self.n_wait += 1
            #     if self.n_wait == 100:
            #         self.done = True

            if self.stage == 'init':
                left_pos = self.world_T_eef_rotate[0][:3, 3]
                left_ori = t3d.quaternions.mat2quat(self.world_T_eef_rotate[0][:3, :3])
                right_pos = self.world_T_eef_rotate[1][:3, 3]
                right_ori = t3d.quaternions.mat2quat(self.world_T_eef_rotate[1][:3, :3])
                if not self.robot.joint_move_left_eef(left_pos, left_ori, randomized=self.randomized) \
                        or not self.robot.joint_move_right_eef(right_pos, right_ori, randomized=self.randomized):
                    self.done = True
                    self.success = False
                self.stage = 'rotate'
                # self.robot.move_left_eef(left_pos, left_ori)
                # self.robot.move_right_eef(right_pos, right_ori)
                # left_curr_pos = robot_obs['left_eef_pos']
                # left_curr_ori = robot_obs['left_eef_ori']
                # right_curr_pos = robot_obs['right_eef_pos']
                # right_curr_ori = robot_obs['right_eef_ori']
                # if np.linalg.norm(left_pos - left_curr_pos) < 0.01 and np.linalg.norm(right_pos - right_curr_pos) < 0.01 \
                #         and np.linalg.norm(left_ori - left_curr_ori) < 0.01 and np.linalg.norm(right_ori - right_curr_ori) < 0.01:
                #     self.stage = 'align'

                self.robot.joint_move_left_eef(left_pos, left_ori)
                self.robot.joint_move_right_eef(right_pos, right_ori)
                self.stage = 'rotate'
            elif self.stage == 'rotate' and (len(self.robot.left_arm_plan) == 0 and len(self.robot.right_arm_plan) == 0):
                left_pos = self.world_T_eef_align[0][:3, 3]
                left_ori = t3d.quaternions.mat2quat(self.world_T_eef_align[0][:3, :3])
                right_pos = self.world_T_eef_align[1][:3, 3]
                right_ori = t3d.quaternions.mat2quat(self.world_T_eef_align[1][:3, :3])
                if not self.robot.joint_move_left_eef(left_pos, left_ori, randomized=self.randomized) \
                        or not self.robot.joint_move_right_eef(right_pos, right_ori, randomized=self.randomized):
                    self.done = True
                    self.success = False
                self.stage = 'align'
                # self.robot.move_left_eef(left_pos, left_ori)
                # self.robot.move_right_eef(right_pos, right_ori)
                # left_curr_pos = robot_obs['left_eef_pos']
                # left_curr_ori = robot_obs['left_eef_ori']
                # right_curr_pos = robot_obs['right_eef_pos']
                # right_curr_ori = robot_obs['right_eef_ori']
                # if np.linalg.norm(left_pos - left_curr_pos) < 0.01 and np.linalg.norm(right_pos - right_curr_pos) < 0.01 \
                #         and np.linalg.norm(left_ori - left_curr_ori) < 0.01 and np.linalg.norm(right_ori - right_curr_ori) < 0.01:
                #     self.stage = 'align'
            elif self.stage == 'align' and (len(self.robot.left_arm_plan) == 0 and len(self.robot.right_arm_plan) == 0):
                left_pos = self.world_T_eef_close[0][:3, 3]#self.world_T_eef_close[0][:3, 3]
                left_ori = t3d.quaternions.mat2quat(self.world_T_eef_close[0][:3, :3])
                right_pos = self.world_T_eef_close[1][:3, 3]# self.world_T_eef_close[1][:3, 3]
                right_ori = t3d.quaternions.mat2quat(self.world_T_eef_close[1][:3, :3])
                if not self.robot.joint_move_left_eef(left_pos, left_ori, randomized=self.randomized) \
                        or not self.robot.joint_move_right_eef(right_pos, right_ori, randomized=self.randomized):
                    self.done = True
                    self.success = False
                self.stage = 'close'
                self.n_wait = 0
                # self.robot.move_left_eef(left_pos, left_ori)
                # self.robot.move_right_eef(right_pos, right_ori)
                # left_curr_pos = robot_obs['left_eef_pos']
                # left_curr_ori = robot_obs['left_eef_ori']
                # right_curr_pos = robot_obs['right_eef_pos']
                # right_curr_ori = robot_obs['right_eef_ori']
                # if np.linalg.norm(left_pos - left_curr_pos) < 0.01 and np.linalg.norm(right_pos - right_curr_pos) < 0.01 \
                #         and np.linalg.norm(left_ori - left_curr_ori) < 0.01 and np.linalg.norm(right_ori - right_curr_ori) < 0.01:
                #     self.stage = 'close'
                #     self.n_wait = 0
        if self.stage == 'close' and (len(self.robot.left_arm_plan) == 0 and len(self.robot.right_arm_plan) == 0):
            self.n_wait += 1
            if self.n_wait >= self.n_wait_threshold:
                self.done = True
        if self.done:
            dists = np.array([np.linalg.norm(world_T_eef[i][:3, 3] - self.world_T_eef_close[i][:3, 3]) for i in range(2)])
            angles = np.array([t3d.axangles.mat2axangle(np.linalg.inv(world_T_eef[i][:3, :3]) @ self.world_T_eef_close[i][:3, :3])[1] for i in range(2)])
            print(dists, angles * 180 / np.pi)
    
    def get_obs(self):
        return {
            'top': self.camera_top.get_obs()['rgb'][:, :, [2, 1, 0]],
            'left': self.camera_left.get_obs()['rgb'][:, :, [2, 1, 0]],
            'right': self.camera_right.get_obs()['rgb'][:, :, [2, 1, 0]],
            'robot': self.robot.get_obs(),
        }
    
    def is_done(self):
        return self.done
    
    def is_success(self):
        return self.success


def pose_to_T(pos, ori):
    return t3d.affines.compose(pos, t3d.quaternions.quat2mat(ori), np.ones(3))

def T_to_pose(T):
    return T[:3, 3], t3d.quaternions.mat2quat(T[:3, :3])


def get_delta(curr_state, next_state):
    world_T_curr_right = t3d.affines.compose(curr_state[:3],
                                             t3d.quaternions.quat2mat(curr_state[3:7]),
                                             np.ones(3))
    world_T_curr_left = t3d.affines.compose(curr_state[7:10],
                                            t3d.quaternions.quat2mat(curr_state[10:14]),
                                            np.ones(3))
    world_T_next_right = t3d.affines.compose(next_state[:3],
                                             t3d.quaternions.quat2mat(next_state[3:7]),
                                             np.ones(3))
    world_T_next_left = t3d.affines.compose(next_state[7:10],
                                            t3d.quaternions.quat2mat(next_state[10:14]),
                                            np.ones(3))
    curr_T_next_right = np.linalg.inv(world_T_curr_right) @ world_T_next_right
    curr_T_next_left = np.linalg.inv(world_T_curr_left) @ world_T_next_left
    delta = np.concatenate([curr_T_next_right[:3, 3], t3d.quaternions.mat2quat(curr_T_next_right[:3, :3]),
                            curr_T_next_left[:3, 3], t3d.quaternions.mat2quat(curr_T_next_left[:3, :3])]).astype(np.float32)
    return delta


def apply_delta(curr_state, delta):
    world_T_curr_right = t3d.affines.compose(curr_state[:3],
                                             t3d.quaternions.quat2mat(curr_state[3:7]),
                                             np.ones(3))
    world_T_curr_left = t3d.affines.compose(curr_state[7:10],
                                            t3d.quaternions.quat2mat(curr_state[10:14]),
                                            np.ones(3))
    curr_T_next_right = t3d.affines.compose(delta[:3],
                                            t3d.quaternions.quat2mat(delta[3:7]),
                                            np.ones(3))
    curr_T_next_left = t3d.affines.compose(delta[7:10],
                                           t3d.quaternions.quat2mat(delta[10:14]),
                                           np.ones(3))
    world_T_next_right = world_T_curr_right @ curr_T_next_right
    world_T_next_left = world_T_curr_left @ curr_T_next_left
    next_state = np.concatenate([world_T_next_right[:3, 3], t3d.quaternions.mat2quat(world_T_next_right[:3, :3]),
                                 world_T_next_left[:3, 3], t3d.quaternions.mat2quat(world_T_next_left[:3, :3])]).astype(np.float32)
    return next_state