from matplotlib import pyplot as plt
import numpy as np
import torch
import pybullet_planning as pp
import transforms3d as t3d
import geometric_peg_in_hole.bc_dataset


def evaluate(env, model, n_evaluations, n_evaluate_steps, device, delta, obs_frames, act_frames, views, seed_start):
    model.eval()
    # input(views)
    n_success = 0
    rgbs = []
    for i_sample in range(n_evaluations):
        env.reset(seed=i_sample+seed_start)
        i_step = 0
        observations = []
        while not env.is_done() and i_step // 24 < n_evaluate_steps and not env.is_success():
            env.update(False)
            if i_step % 24 == 0:
                observations.append({})
                observations[-1]['top'] = env.camera_top.get_obs()['rgb'][:, :, [0, 1, 2]].transpose(2, 0, 1)
                observations[-1]['left_cam'] = env.camera_left.get_obs()['rgb'][:, :, [0, 1, 2]].transpose(2, 0, 1)
                observations[-1]['right_cam'] = env.camera_right.get_obs()['rgb'][:, :, [0, 1, 2]].transpose(2, 0, 1)
                observations[-1].update(env.robot.get_obs())

                images = []
                robot_states = []
                for i_frame in obs_frames:
                    i_curr = max(0, min(len(observations) - 1 + i_frame, len(observations) - 1))
                    images.append([])
                    for view in views:
                        if view == 'left_crop':
                            images[-1].append(observations[i_curr]['top'][:, 65:235, 50:220])
                        elif view == 'right_crop':
                            images[-1].append(observations[i_curr]['top'][:, 110:280, 280:450])
                        else:
                            images[-1].append(observations[i_curr][view])
                    # robot_states.append(lib.bc_dataset.get_action(i_curr - 1, observations, model.rotation_type, False))
                    # right_action = np.concatenate([observations[i_curr]['right_eef_pos'],
                    #                                observations[i_curr]['right_eef_ori'],])
                    # left_action = np.concatenate([observations[i_curr]['left_eef_pos'],
                    #                               observations[i_curr]['left_eef_ori'],])
                    if model.rotation_type == 'quat':
                        right_action = np.concatenate([observations[i_curr]['right_eef_pos'],
                                                    observations[i_curr]['right_eef_ori']])
                        left_action = np.concatenate([observations[i_curr]['left_eef_pos'],
                                                    observations[i_curr]['left_eef_ori']])
                    elif model.rotation_type == '6d':
                        right_action = np.concatenate([observations[i_curr]['right_eef_pos'],
                                                        np.reshape(t3d.quaternions.quat2mat(observations[i_curr]['right_eef_ori'])[:3, :2], -1, order='F')])
                        left_action = np.concatenate([observations[i_curr]['left_eef_pos'],
                                                        np.reshape(t3d.quaternions.quat2mat(observations[i_curr]['left_eef_ori'])[:3, :2], -1, order='F')])
                    robot_states.append(np.concatenate([right_action, left_action]))
                robot_states = np.stack(robot_states, axis=0).astype(np.float32)
                images = np.stack([np.stack(frame) for frame in images]).astype(np.float32) / 255
                images = torch.from_numpy(images).to(device)[None, ...]
                robot_states = torch.from_numpy(robot_states).to(device)[None, ...]

                with torch.no_grad():
                    actions = model.get_action(images, robot_states)
                    actions = actions.cpu().numpy()[0, 0]
                if delta:
                    robot_states = robot_states.cpu().numpy()
                    actions = apply_delta(robot_states[0, -1], actions)
                
                if not env.robot.joint_move_right_eef(actions[:3], actions[3:7]) or not env.robot.joint_move_left_eef(actions[7:10], actions[10:14]):
                    print('failed move')
                    break

            pp.step_simulation()
            i_step += 1
        
        rgbs.append([])
        if len(rgbs) <= 10:
            for obs in observations:
                rgbs[-1].append(obs['top'])
        if env.is_success():
            print(f'Success {i_sample} - {i_step // 24} steps')
            success_frame = np.zeros_like(obs['top'])
            success_frame[2, :, :] = 255
            rgbs[-1].append(success_frame)
            n_success += 1
        else:
            failure_frame = np.zeros_like(obs['top'])
            failure_frame[0, :, :] = 255
            rgbs[-1].append(failure_frame)
            print(f'Failure {i_sample} - {i_step // 24} steps')
    pp.disconnect()
    return n_success / n_evaluations, rgbs

def apply_delta(state, action):
    world_T_right = t3d.affines.compose(state[:3], t3d.quaternions.quat2mat(state[3:7]),
                                             np.ones(3))
    world_T_left = t3d.affines.compose(state[7:10], t3d.quaternions.quat2mat(state[10:14]),
                                            np.ones(3))
    right_T_right_next = t3d.affines.compose(action[:3], t3d.quaternions.quat2mat(action[3:7]),
                                                    np.ones(3))
    left_T_left_next = t3d.affines.compose(action[7:10], t3d.quaternions.quat2mat(action[10:14]),
                                                    np.ones(3))
    world_T_right_next = world_T_right @ right_T_right_next
    world_T_left_next = world_T_left @ left_T_left_next
    right_pos = world_T_right_next[:3, 3]
    right_ori = t3d.quaternions.mat2quat(world_T_right_next[:3, :3])
    left_pos = world_T_left_next[:3, 3]
    left_ori = t3d.quaternions.mat2quat(world_T_left_next[:3, :3])

    right_pos, right_ori, _, _ = t3d.affines.decompose(world_T_right_next)
    left_pos, left_ori, _, _ = t3d.affines.decompose(world_T_left_next)
    right_ori = t3d.quaternions.mat2quat(right_ori)
    left_ori = t3d.quaternions.mat2quat(left_ori)

    return np.concatenate([right_pos, right_ori, left_pos, left_ori])