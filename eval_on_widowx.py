

import torch
import hydra, json
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./conf", config_name="bridge-64")
def my_main(cfg: DictConfig):
    # Define the path to the saved entire model
    load_path = 'miniGRP.pth'

    # Load the entire model
    model = torch.load(load_path)

    # Set the model to evaluation mode if you intend to use it for inference
    # loaded_model.eval()

    print("Entire model loaded successfully.")



    while not (done or truncated or (t > timeLimit)):
        # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
        # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
        image = get_image_from_maniskill2_obs_dict(env_unwrapped, obs)
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
        obs, reward, done, truncated, info = env.step(action)
        reward = -np.linalg.norm(info["eof_to_obj1_diff"])
        frames.append(image)
        rewards.append(reward)
        t=t+1