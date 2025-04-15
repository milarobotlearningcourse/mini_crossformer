#!/usr/bin/env python3

import torch
import hydra, json
from omegaconf import DictConfig, OmegaConf


from absl import app, flags, logging

flags.DEFINE_string("ip", "norris", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")
# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/C920/image_raw"}]
FIXED_STD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

# bridge_data_robot imports
from widowx_envs.widowx_env_service import WidowXClient, WidowXStatus, WidowXConfigs
from utils import state_to_eep, stack_obs


@hydra.main(config_path="./conf", config_name="bridge-64")
def my_main(cfg: DictConfig):

    assert isinstance(FLAGS.initial_eep, list)
    initial_eep = [float(e) for e in FLAGS.initial_eep]
    start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])

    # set up environment
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    env_params["state_state"] = list(start_state)
    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    widowx_client.init(env_params, image_size=FLAGS.im_size)


    # Define the path to the saved entire model
    load_path = 'miniGRP.pth'

    # Load the entire model
    model = torch.load(load_path)

    # Set the model to evaluation mode if you intend to use it for inference
    # loaded_model.eval()

    print("Entire model loaded successfully.")

    # reset env
    widowx_client.reset()
    time.sleep(2.5)

    while not (done or (t > 100)):
        # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
        # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
        image = widowx_client.get_observation()
        last_tstep = time.time()
        actions = get_action(obs, goal_obs)

        # perform environment step
        # image = get_image_from_maniskill2_obs_dict(env_unwrapped, obs)
        image = image[:,:,:3] ## Remove last dimension of image color
        
        action, loss = model.forward(torch.tensor(np.array([encode_state(resize_state(image))])).to(device)
                            # ,torch.tensor(txt_goal, dtype=torch.float).to(device) ## There can be issues here if th text is shorter than any example in the dataset
                            ,torch.tensor(txt_goal, dtype=torch.long).to(device) ## There can be issues here if th text is shorter than any example in the dataset
                            ,torch.tensor(np.array([encode_state(resize_state(image))])).to(device) ## Not the correct goal image... Should mask this.
                            )
        # action = env.action_space.sample() # replace this with your policy inference
        if cfg.load_action_bounds:
            action = decode_action(action.cpu().detach().numpy()[0]) ## Add in the gripper close action
        else:
            action = np.concatenate((decode_action(action.cpu().detach().numpy()[0]), [0]), axis = -1) ## Add in the gripper close action
        widowx_client.step_action(action, blocking=FLAGS.blocking)
        # obs, reward, done, truncated, info = env.step(action)
        # reward = -np.linalg.norm(info["eof_to_obj1_diff"])
        # frames.append(image)
        # rewards.append(reward)
        t=t+1