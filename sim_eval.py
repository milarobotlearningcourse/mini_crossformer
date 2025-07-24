

def get_text_tokens(cfg, tokenizer, text_model, goal):
    """
    Get the text tokens for the goal.
    """
    if cfg.dataset.encode_with_t5:
        goal_ = np.zeros((cfg.max_block_size, cfg.n_embd))
        input_ids = tokenizer(goal, return_tensors="pt").input_ids
        goal_t = text_model.encoder(input_ids).last_hidden_state.detach().cpu().numpy() ## Get the goal embedding
        goal_[:len(goal_t[0]), :] = goal_t[0][:cfg.max_block_size] ## Overwrite just the zeros up to the size of this vector, smaller vectors will have < max_block_size
    else:
        goal_ = " " * cfg.max_block_size
        goal_ = goal[:cfg.max_block_size] + goal_[len(goal):cfg.max_block_size]
    return [goal_]

def eval_model_in_sim(cfg, model, device, log_dir, env, env_unwrapped, buffer,
                      wandb, iter_, tokenizer=None, text_model=None):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    print("Evaluating model in sim environment")
    from collections import deque
    from einops import rearrange

    rewards = []
    for j in range(cfg.sim.eval_episodes): ## Better to eval over a few different goal configurations
        obs, reset_info = env.reset()
        obs_hist = deque(maxlen=3)
        obs_hist.append(obs['image']['base_camera']["rgb"])
        obs_hist.append(obs['image']['base_camera']["rgb"])
        obs_hist.append(obs['image']['base_camera']["rgb"])
        instruction = env_unwrapped.get_language_instruction()
        print("Reset info", reset_info)
        print("Instruction", instruction)
        frames = []
        done, truncated, timeLimit, t = False, False, 100, 0
        txt_goal = get_text_tokens(cfg, tokenizer, text_model, instruction)
        while not (done or truncated or (t > timeLimit)):
            # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
            # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
            image = get_image_from_maniskill2_obs_dict(env_unwrapped, obs)
            image = image[:,:,:3] ## Remove last dimension of image color
            
            obs_hist.append(obs['image']['base_camera']["rgb"]) ## Add the new observation to the history buffer
            # obs = [obs_["image"] for obs_ in obs] # obs is a list of dicts
            image = np.stack(obs_hist, axis=-1)  # stack along the last dimension
            image = rearrange(image, 'h w c t -> h w (c t)')  # add batch dimension
            
            action, loss = model.forward(torch.tensor(np.array([buffer._encode_state(buffer._resize_state(image))])).to(device)
                                # ,torch.tensor(txt_goal, dtype=torch.float).to(device) ## There can be issues here if th text is shorter than any example in the dataset
                                ,torch.tensor(txt_goal, dtype=torch.long).to(device) ## There can be issues here if th text is shorter than any example in the dataset
                                ,torch.tensor(np.array([buffer._encode_state(buffer._resize_state(image[:,:,:3]))])).to(device) ## Not the correct goal image... Should mask this.
                                )
            
            action = buffer._decode_action(action[0,:7]).cpu().detach().numpy() ## Add in the gripper close action
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

import gymnasium as gym
# --- History Stacking Wrapper ---
class DictWrapper(gym.ObservationWrapper):
    # from gymnasium.spaces import Box
    """
    A wrapper that grabs the observation from a specific key in the dictionary.
    """
    def __init__(self, env, obs_key=""):
        # gym.Wrapper.__init__(self, env)
        self.env = env
        self.observation_space = gym.spaces.Box( 
            low=0,
            high=255,
            shape=(128,128,3),  # Assuming the observation is an image of size 128x128 with 3 color channels
            dtype=np.uint8)
        self._obs_key = obs_key

    def observation(self, observation):
        """
        This method is called by the gym.ObservationWrapper after the environment's
        step or reset methods return an observation.
        """
        # Add the new observation to the history buffer
        return observation[self._obs_key]
    
    def step(self, action):
        """
        Step the environment and return the observation from the specified key.
        """
        obs, reward, done, info = self.env.step(action) ## LIBERO does not return truncated
        return obs[self._obs_key][::-1, :, :], reward, done, False, obs ## Not sure why the image was upside down.

    def reset(self, **kwargs):
        """
        Reset the environment and return the observation from the specified key.
        """
        obs = self.env.reset()
        return obs[self._obs_key][::-1, :, :], {}

def eval_libero(buffer, model, device, cfg, iter_=0, log_dir="./", 
                tokenizer=None, text_model=None, wandb=None):
        # cfg, model, device, log_dir, env, env_unwrapped, buffer,
        #               wandb, iter_, tokenizer=None, text_model=None):
    
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv, DenseRewardEnv
    import os
    from libero.libero.utils import get_libero_path
    from gymnasium.wrappers import FrameStackObservation
    from einops import rearrange


    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_90" # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()

    # retrieve a specific task
    tasks = cfg.sim.eval_tasks
    for task_id in tasks:
        task = task_suite.get_task(task_id)
        task_name = task.name
        instruction = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
            f"language instruction is {instruction}, and the bddl file is {task_bddl_file}")

        # step over the environment
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 128,
            "camera_widths": 128
        }
        env = DenseRewardEnv(**env_args)
        env.seed(0)
        init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
        init_state_id = 0
        env.set_init_state(init_states[init_state_id])
        env = FrameStackObservation(DictWrapper(env, obs_key="agentview_image"), cfg.policy.obs_stacking) ## Stacking the observations
        obs, info = env.reset()

        txt_goal = get_text_tokens(cfg, tokenizer, text_model, instruction)
        image_goal = obs.reshape((128, 128, 3*cfg.policy.obs_stacking))[:,:,:3] ## Assuming the observation is an image of size 128x128 with 3 color channels
        frames = []
        rewards = []
        infos = []
        for step_ in range(250):
            ## Reshape the image to the correct size and stack the hostory on the last channel dimension
            image = obs[0]
            # obs = obs.reshape((128, 128, 3*cfg.policy.obs_stacking)) ## Assuming the observation is an image of size 128x128 with 3 color channels  
            obs = rearrange(obs, 't h w c -> h w (t c)', c=3, t=cfg.policy.obs_stacking) ## Rearranging the image to have the stacked history in the last channel dimension
            # image = obs[:,:,:3] ## Remove the last dimension of the image color
            action, loss = model.forward(torch.tensor(np.array([buffer._encode_state(buffer._resize_state(obs))])).to(device)
                        ,torch.tensor(txt_goal, dtype=torch.float).to(device) ## There can be issues here if th text is shorter than any example in the dataset
                        # ,torch.tensor(txt_goal, dtype=torch.long).to(device) ## There can be issues here if th text is shorter than any example in the dataset
                        ,torch.tensor(np.array([buffer._encode_state(buffer._resize_state(image_goal))])).to(device) ## Not the correct goal image... Should mask this.
                        )

            action = buffer._decode_action(action[0,:7]).cpu().detach().numpy() ## Add in the gripper close action
            frames.append(image)
            x = env.step(action)
            obs, reward, done, truncated, info = x
            rewards.append(reward)
            infos.append(info)
            if done:
                print("Episode finished after {} timesteps".format(step_))
                break

        print(f"avg reward {np.mean(rewards):.8f}")
        detail_name = "akita_black_bowl_1_to_robot0_eef_pos"
        print({"avg "+detail_name+" for task "+str(task_id): np.mean([np.linalg.norm(info[detail_name]) for info in infos])}) if detail_name in infos[0].keys() else " "
        detail_name = "butter_1_to_robot0_eef_pos"
        print({"avg "+detail_name+" for task "+str(task_id): np.mean([np.linalg.norm(info[detail_name]) for info in infos])}) if detail_name in infos[0].keys() else " "
        detail_name = "butter_2_to_robot0_eef_pos"
        print({"avg "+detail_name+" for task "+str(task_id): np.mean([np.linalg.norm(info[detail_name]) for info in infos])}) if detail_name in infos[0].keys() else " "
        detail_name = "chocolate_pudding_1_to_robot0_eef_pos"
        print({"avg "+detail_name+" for task "+str(task_id): np.mean([np.linalg.norm(info[detail_name]) for info in infos])}) if detail_name in infos[0].keys() else " "
        if not cfg.testing:
            wandb.log({"avg reward_"+str(task_id): np.mean(rewards)})
            detail_name = "akita_black_bowl_1_to_robot0_eef_pos"
            wandb.log({"avg "+detail_name+" for task "+str(task_id): np.mean([np.linalg.norm(info[detail_name]) for info in infos])}) if detail_name in infos[0].keys() else " "
            detail_name = "butter_1_to_robot0_eef_pos"
            wandb.log({"avg "+detail_name+" for task "+str(task_id): np.mean([np.linalg.norm(info[detail_name]) for info in infos])}) if detail_name in infos[0].keys() else " "
            detail_name = "butter_2_to_robot0_eef_pos"
            wandb.log({"avg "+detail_name+" for task "+str(task_id): np.mean([np.linalg.norm(info[detail_name]) for info in infos])}) if detail_name in infos[0].keys() else " "
            detail_name = "chocolate_pudding_1_to_robot0_eef_pos"
            wandb.log({"avg "+detail_name+" for task "+str(task_id): np.mean([np.linalg.norm(info[detail_name]) for info in infos])}) if detail_name in infos[0].keys() else " "
        import moviepy.editor as mpy
        clip = mpy.ImageSequenceClip(list(frames), fps=20)
        clip.write_videofile(log_dir+"/sim-libero-90-"+str(task_id)+"-"+str(iter_)+".mp4", fps=20)
        if not cfg.testing:
            wandb.log({"example": wandb.Video(log_dir+"/sim-libero-90-"+str(task_id)+"-"+str(iter_)+".mp4")})
        env.close()

import hydra, json
from omegaconf import DictConfig, OmegaConf
from mini_grp2 import *

@hydra.main(config_path="./conf", config_name="libero-simpleEnv-64pix")
def my_main(cfg: DictConfig):
    import tensorflow_datasets as tfds  
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
    model_._cgf = cfg

    if cfg.dataset.encode_with_t5: ## Load T5 model
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        tokenizer = T5Tokenizer.from_pretrained(cfg.dataset.t5_version)
        text_model = T5ForConditionalGeneration.from_pretrained(cfg.dataset.t5_version)
    
    if "simple_env" in cfg.simEval:
        import simpler_env
        task_name = "widowx_carrot_on_plate"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
        if 'env' in locals():
            print("Closing existing env")
            env.close()
            del env
        env = simpler_env.make(task_name)
        env_unwrapped = env.env.env.env ## Updated gymnasium wrapper adds lots of wrappers.
        results = eval_model_in_sim(cfg, model_.to(cfg.device), device=cfg.device, log_dir="./",
                                env=env, env_unwrapped=env_unwrapped,
                                buffer=cBuffer, wandb=None, iter_=0, tokenizer=tokenizer, text_model=text_model)

    if "libero" in cfg.simEval:
        results = eval_libero(cBuffer, model_.to(cfg.device), device=cfg.device, cfg=cfg,
                          iter_=0, tokenizer=tokenizer, text_model=text_model, wandb=None)
    # print("results:", results)
    # cbuffer.save(cfg.dataset.to_name)


if __name__ == "__main__":
    results = my_main()
    print("results:", results)