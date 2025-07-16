


def eval_model_in_sim(cfg, model, device, log_dir, env, env_unwrapped, buffer,
                      wandb, iter_, tokenizer=None, text_model=None):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    print("Evaluating model in sim environment")

    rewards = []
    for j in range(cfg.sim.eval_episodes): ## Better to eval over a few different goal configurations
        obs, reset_info = env.reset()
        instruction = env_unwrapped.get_language_instruction()
        print("Reset info", reset_info)
        print("Instruction", instruction)
        frames = []
        done, truncated, timeLimit, t = False, False, 100, 0
        if cfg.dataset.encode_with_t5:
            input_ids = tokenizer(instruction, return_tensors="pt").input_ids
            txt_goal_ = np.array([text_model.encoder(input_ids).last_hidden_state.detach().numpy()[0][:cfg.max_block_size]]) ## All just to trim the tensor down to the min size in the dataset
            txt_goal = np.zeros((cfg.max_block_size,cfg.n_embd))
            txt_goal[:len(txt_goal_[0]), :] = txt_goal_
            txt_goal = [txt_goal]
        else:
            instruction = instruction[:cfg.max_block_size] + str(" " * cfg.max_block_size)[len(instruction):cfg.max_block_size] ## padding the string length to block size.
            txt_goal = np.array([buffer._encode_txt(instruction)[:cfg.max_block_size]])
        while not (done or truncated or (t > timeLimit)):
            # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
            # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
            image = get_image_from_maniskill2_obs_dict(env_unwrapped, obs)
            image = image[:,:,:3] ## Remove last dimension of image color
            
            action, loss = model.forward(torch.tensor(np.array([buffer._encode_state(buffer._resize_state(image))])).to(device)
                                # ,torch.tensor(txt_goal, dtype=torch.float).to(device) ## There can be issues here if th text is shorter than any example in the dataset
                                ,torch.tensor(txt_goal, dtype=torch.long).to(device) ## There can be issues here if th text is shorter than any example in the dataset
                                ,torch.tensor(np.array([buffer._encode_state(buffer._resize_state(image))])).to(device) ## Not the correct goal image... Should mask this.
                                )
            
            action = buffer._decode_action(action.cpu().detach().numpy()[0]) ## Add in the gripper close action
            obs, reward, done, truncated, info = env.step(action)
            reward = -np.linalg.norm(info["eof_to_obj1_diff"])
            frames.append(image)
            rewards.append(reward)
            t=t+1
    
    episode_stats = info.get('episode_stats', {})
    print("Episode stats", episode_stats)
    print(f"avg reward {np.mean(rewards):.8f}")
    if not cfg.testing:
        wandb.log({"avg reward": np.mean(rewards)})
    import moviepy.editor as mpy
    clip = mpy.ImageSequenceClip(list(frames), fps=20)
    clip.write_videofile(log_dir+"/sim-env-"+str(iter_)+".mp4", fps=20)
    if not cfg.testing:
        wandb.log({"example": wandb.Video(log_dir+"/sim-env-"+str(iter_)+".mp4")})


def eval_libero(buffer, model, device, cfg, iter_=0, log_dir="./", 
                tokenizer=None, text_model=None, wandb=None):
        # cfg, model, device, log_dir, env, env_unwrapped, buffer,
        #               wandb, iter_, tokenizer=None, text_model=None):
    
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    import os
    from libero.libero.utils import get_libero_path


    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_90" # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()

    # retrieve a specific task
    task_id = 0
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
        f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    obs = env.reset()
    init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
    init_state_id = 0
    env.set_init_state(init_states[init_state_id])

    txt_goal = np.array([buffer._encode_txt(task_description)[:cfg.max_block_size]])
    dummy_action = [0.] * 7
    # image = obs["agentview_image"]
    frames = []
    rewards = []
    for step in range(100):
        image = obs["agentview_image"]
        action, loss = model.forward(torch.tensor(np.array([buffer._encode_state(buffer._resize_state(image))])).to(device)
                    # ,torch.tensor(txt_goal, dtype=torch.float).to(device) ## There can be issues here if th text is shorter than any example in the dataset
                    ,torch.tensor(txt_goal, dtype=torch.long).to(device) ## There can be issues here if th text is shorter than any example in the dataset
                    ,torch.tensor(np.array([buffer._encode_state(buffer._resize_state(image))])).to(device) ## Not the correct goal image... Should mask this.
                    )

        action = buffer._decode_action(action).cpu().detach().numpy()[0] ## Add in the gripper close action
        frames.append(image)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)

    print(f"avg reward {np.mean(rewards):.8f}")
    if not cfg.testing:
        wandb.log({"avg reward": np.mean(rewards)})
    import moviepy.editor as mpy
    clip = mpy.ImageSequenceClip(list(frames), fps=20)
    clip.write_videofile(log_dir+"/sim-env-"+str(iter_)+".mp4", fps=20)
    if not cfg.testing:
        wandb.log({"example": wandb.Video(log_dir+"/sim-env-"+str(iter_)+".mp4")})
    env.close()

import hydra, json
from omegaconf import DictConfig, OmegaConf
from mini_grp2 import *

@hydra.main(config_path="./conf", config_name="libero-64pix")
def my_main(cfg: DictConfig):
    import tensorflow_datasets as tfds
    import numpy as np
    from tqdm import tqdm, trange    
    from mini_shuffel_buffer import CircularBuffer
    import torch
    # ------------
    # Train and test splits
    # Loading data
    # create RLDS dataset builder
    cfg.dataset.load_dataset = "skip"
    cBuffer = CircularBuffer(cfg.dataset.buffer_size, cfg)
    model = GRP(cfg)
    model_ = torch.load("/home/mila/g/glen.berseth/playground/mini-grp/miniGRP.pth")

    results = eval_libero(cBuffer, model_.to(cfg.device), device=cfg.device, cfg=cfg)
    # cbuffer.save(cfg.dataset.to_name)


if __name__ == "__main__":
    results = my_main()
    print("results:", results)