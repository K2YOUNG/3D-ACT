import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

import yaml
import time

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data, generate_points # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def set_ur5(num_cameras, state_dim, reset_joints):
    import rospy
    from gellosoftware_BI.gello.env import RobotEnv
    from gellosoftware_BI.gello.zmq_core.robot_node import ZMQClientRobot
    from gellosoftware_BI.gello.cameras.realsense_camera import LogitechCamera, RealSenseCamera, RealSenseCameraRos, get_device_ids
    if num_cameras >= 2:
        if state_dim == 14:
            ## 먼저 realsense수 체크
            ids = get_device_ids()
            camera_clients = {}
            for id in ids:
                if id == '033422070567': #left camera
                    camera_clients["wrist_left"] = RealSenseCamera(device_id=id)
                elif id == '021222071327':
                    camera_clients["wrist_right"] = RealSenseCamera(device_id=id)
                        
            camera_clients["base"] = LogitechCamera(device_id='/dev/frontcam')
        else:
            print("DUAL CAMERA!!!")
            camera_clients = {
                # you can optionally add camera nodes here for imitation learning purposes
                # "wrist": RealSenseCamera(),
                "base": LogitechCamera(device_id='/dev/frontcam'),
                "arti": RealSenseCameraRos(topic='camera')
            }
            
        print("FINISH")
    else:
        # Going to use more than two cameras
        # Skipping to implement the single-camera situation
        # TODO : Implement single-camera situation
        pass

    robot_client = ZMQClientRobot(port=6001, host="127.0.0.1")
    env = RobotEnv(robot_client, control_rate_hz=50, camera_dict=camera_clients)

    if state_dim == 14:
        # dynamixel control box port map (to distinguish left and right gello)

        ## 1119 세팅!! TODO 바꿔야함
        reset_joints_left = np.deg2rad([149, -58, -134, -77, 87, -45, 0])
        reset_joints_right = np.deg2rad([-143, -112, 127, -104, -93, 45, 0])
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
    else:
        it = 1 # right robot
        reset_joints = reset_joints
        curr_joints = env.get_obs()["joint_positions"]
        if reset_joints.shape == curr_joints.shape:
            max_delta = (np.abs(curr_joints - reset_joints)).max()
            steps = min(int(max_delta / 0.01), 100)

            for jnt in np.linspace(curr_joints, reset_joints, steps):
                env.step(jnt)
                time.sleep(0.001)
        
    # going to start position
    print("Going to start position")
    scan_cam = camera_clients['arti']

    time.sleep(1)

    return env, scan_cam

def main(args):
    with open(args["config_path"]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    set_seed(1)
    # command line parameters
    is_eval = config['evaluation']['is_eval']
    ckpt_dir = config['model']['ckpt_dir']
    policy_class = config['model']['policy_class']
    # onscreen_render = args['onscreen_render']
    task_name = config['data']['task_name']
    batch_size_train = config['train']['batch_size']
    batch_size_val = config['train']['batch_size']
    num_epochs = config['train']['num_epochs']

    is_sim = config['data']['is_sim']
    dataset_dir = config['data']['dataset_dir']
    num_episodes = config['data']['num_episodes']
    episode_len = config['data']['episode_len']
    use_color = config['data']['use_color']
    use_pointcloud = config['data']['use_pointcloud']
    num_points = config['data']['num_points']
    num_cameras = config['data']['num_cameras']
    reset_joints = config['robot']['reset_joints']

    # fixed parameters
    state_dim = config['model']['state_dim']
    lr_backbone = config['model']['lr_backbone']
    backbone = config['model']['backbone']
    if policy_class == 'ACT':
        enc_layers = config['model']['ACT']['enc_layers']
        dec_layers = config['model']['ACT']['dec_layers']
        nheads = config['model']['ACT']['nheads']
        policy_config = {'lr': config['train']['lr'],
                         'num_queries': config['model']['ACT']['chunk_size'],
                         'kl_weight': config['train']['kl_weight'],
                         'hidden_dim': config['model']['ACT']['hidden_dim'],
                         'dim_feedforward': config['model']['ACT']['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                        #  'camera_names': camera_names,
                         'use_pointcloud': use_pointcloud
                         }
    elif policy_class == 'CNNMLP':
        raise NotImplementedError("DID not implement CNNMLP - USE ACT")
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                        #  'camera_names': camera_names,
                         }
    else:
        raise NotImplementedError("USE ACT")

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': config['train']['lr'],
        'policy_class': policy_class,
        # 'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': config['train']['seed'],
        'temporal_agg': config['evaluation']['temporal_agg'],
        # 'camera_names': camera_names,
        'real_robot': not is_sim
    }

    if not is_sim and is_eval:
        env, scan_cam = set_ur5(num_cameras, state_dim, reset_joints)
        config['env'] = env
    else:
        # TODO : Implement evaluation code for simulation
        pass

    # ======= EVALUATION
    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, scan_cam, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()


    # ======= TRAIN
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, None, batch_size_train, batch_size_val, num_points, use_color)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(obs):
    curr_images = []
    for name in obs.keys():
        if name.endswith("rgb"):
            curr_image = rearrange(obs[name], 'h w c -> c h w')
            curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, scan_cam, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    # onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    # camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    # onscreen_cam = 'angle'
    env = config['env']

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        # from aloha_scripts.robot_utils import move_grippers # requires aloha
        # from aloha_scripts.real_env import make_real_env # requires aloha
        # env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        # TODO : Implement evaluation code for simulation
        raise NotImplementedError("DIDN'T implemented evaluation codes for simulations")
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        # ts = env.reset()

        # ### onscreen render
        # if onscreen_render:
        #     ax = plt.subplot()
        #     plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
        #     plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                tic = time.time()
                ### update onscreen render and wait for DT
                # if onscreen_render:
                #     image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                #     plt_img.set_data(image)
                #     plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = env.get_obs()
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})

                qpos_numpy = np.array(obs['joint_positions'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                # curr_image = get_image(obs)

                rgb, depth = scan_cam.read()
                points = generate_points(rgb, depth, scan_cam.camera_matrix)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, points)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raise NotImplementedError("DID not implement CNNMLP - USE ACT")
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(0)
                print("ELAPSED TIME", time.time() - tic)

            plt.close()
        if real_robot:
            # move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    points_data, qpos_data, action_data, is_pad = data
    points_data, qpos_data, action_data, is_pad = points_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, points_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']           # ACT or CNNMLP
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default=os.path.join(os.getcwd(), "act", "config/config.yaml"), type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    parser.add_argument("--num_points", type=int, help='Number of points to include in a single sample', required=True)

    parser.add_argument("--dataset_dir", default="data", type=str)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    parser.add_argument("--use_color", action="store_true")
    parser.add_argument("--use_pointcloud", action="store_true")

    
    main(vars(parser.parse_args()))
