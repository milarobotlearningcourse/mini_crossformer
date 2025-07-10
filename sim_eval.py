


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