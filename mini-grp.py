import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm, trange
import cv2


# data loading
def get_batch_grp(split, dataset, batch_size):
    # generate a small batch of inputs x and targets y
    data = dataset['train'] if split == 'train' else dataset['test']
    ix = np.random.randint(int(len(data["img"])), size=(batch_size,))
    x = torch.tensor(data["img"][ix], dtype=torch.float)
    x_goal = torch.tensor(data["goal"][ix], dtype=torch.long)
    x_goal_img = torch.tensor(data["goal_img"][ix], dtype=torch.float)
    y = torch.tensor(data["action"][ix], dtype=torch.float)
    return x, x_goal, x_goal_img, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X, x_goal, x_goal_img, Y = get_batch_grp(split, model._dataset, model._cfg.batch_size)
            logits, loss = model(X, x_goal, x_goal_img, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_patches_fast(images):
    from einops import rearrange
    batch_size, channels, height, width = images.shape
    patch_size = height // 8 ## n_patches = 8

    patches = rearrange(images, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
    return patches

def calc_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

## This is an encoder head (full attention)
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B,T,C = x.shape
        # TODO: 
        ## Provide the block masking
        pass
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        ### Block masked attention
        wei = wei.masked_fill(mask == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x,)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class GRP(nn.Module):
  def __init__(self, dataset, cfg, mlp_ratio=4):
    super(GRP, self).__init__()
    self._dataset = dataset
    self._cfg = cfg
    # TODO: 
    ## Provide the logic for the GRP network

    # 4) Transformer encoder blocks

    # 5) Classification MLPk

  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, images, goals_txt, goal_imgs, targets=None):
    # Dividing images into patches
    n, c, h, w = images.shape
    B, T = goals_txt.shape
    # TODO: 
    ## Provide the logic to produce the output and loss for the GRP
    
    # Map the vector corresponding to each patch to the hidden size dimension

    # Adding classification and goal_img tokens to the tokens

    # Adding positional embedding

    # Compute blocked masks

    # Transformer Blocks

    # Getting the classification token only

    # Compute output and loss
    return (out, loss)

import hydra, json
from omegaconf import DictConfig, OmegaConf

# @hydra.main(config_path="conf", config_name="grp-mini")
@hydra.main(config_path="./conf", config_name="bridge-64-light")
def my_main(cfg: DictConfig):
    torch.manual_seed(cfg.r_seed)
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print ("cfg:", OmegaConf.to_yaml(cfg))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    cfg.device = device
    from datasets import load_dataset, load_from_disk

    dataset = load_dataset(cfg.dataset.to_name, split='train')
    print('Features:', dataset.features)

    dataset_tmp = {
        "img": np.array(dataset["img"]),
        "action": np.concatenate((np.array(dataset["action"]) 
                                ,np.array(dataset["rotation_delta"])
                                ,np.array(dataset["open_gripper"])
                                ), axis=1),
        "goal_img": np.array(dataset["goal_img"]),
        "goal": dataset["goal"]
    }
    shortest_text_len = min([len(txt) for txt in dataset["goal"]])
    cfg.block_size = shortest_text_len

    # here are all the unique characters that occur in this text
    chars = sorted(list(set([item for row in dataset_tmp["goal"] for item in row]))) ## Flatten to a long string
    cfg.vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode_txt = lambda s: [stoi[c] for c in s] # text encoder to tokens: 
    decode_txy = lambda l: ''.join([itos[i] for i in l]) # token decoder to text: 
    print("vocab_size:", cfg.vocab_size)
    print("example text encode:", encode_txt(dataset_tmp["goal"][0]))

    # TODO: 
    ## Provide the logic for the GRP policy for discretized or continuous actions

    ## Get the actions and encode them to map to [-1, 1]
    encode_state = lambda af:   ((af/(255.0)*2.0)-1.0).astype(np.float32) # encoder: take a float, output an integer
    resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state

    n = int(0.9*len(dataset_tmp["img"])) # first 90% will be train, rest val
    dataset_tmp = { 
        "train":
            {
            "img": torch.tensor(encode_state(dataset_tmp["img"][:n])).to(device),
            "action": torch.tensor(encode_action(dataset_tmp["action"][:n]), dtype=torch.float).to(device),            
            "goal_img": torch.tensor(encode_state(dataset_tmp["goal_img"][:n])).to(device),
            "goal": torch.tensor([encode_txt(goal[:cfg.block_size]) for goal in dataset_tmp["goal"][:n]]).to(device)
            },
        "test": 
        {
            "img": torch.tensor(encode_state(dataset_tmp["img"][n:])).to(device),
            "action": torch.tensor(encode_action(dataset_tmp["action"][n:]), dtype=torch.float).to(device),            
            "goal_img": torch.tensor(encode_state(dataset_tmp["goal_img"][n:])).to(device),
            "goal": torch.tensor([encode_txt(goal[:cfg.block_size]) for goal in dataset_tmp["goal"][n:]]).to(device)
        }
    }

    if not cfg.testing:
        import wandb
        # start a new wandb run to track this script
        wandb.init(
            project=cfg.experiment.project,
            # track hyperparameters and run metadata
            config= OmegaConf.to_container(cfg)
        )
        wandb.run.log_code(".")
    model = GRP(dataset_tmp, cfg)
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    import torch.optim.lr_scheduler as lr_scheduler
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.max_iters)

    if cfg.simEval:
        import simpler_env
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
        task_name = "widowx_carrot_on_plate"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
        if 'env' in locals():
            print("Closing existing env")
            env.close()
            del env
        env = simpler_env.make(task_name)
        env_unwrapped = env.env.env.env ## Updated gymnasium wrapper adds lots of wrappers.

    for iter in range(cfg.max_iters):

        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if not cfg.testing:
                wandb.log({"train loss": losses['train'], "val loss": losses['val']})

            if cfg.simEval and (iter % cfg.eval_vid_iters == 0): ## Do this eval infrequently because it takes a fiar bit of compute
                rewards = []
                for j in range(cfg.sim.eval_episodes): ## Better to eval over a few different goal configurations
                    obs, reset_info = env.reset()
                    instruction = env_unwrapped.get_language_instruction()
                    print("Reset info", reset_info)
                    print("Instruction", instruction)
                    frames = []
                    done, truncated, timeLimit, t = False, False, 100, 0
                    while not (done or truncated or (t > timeLimit)):
                        # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
                        # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
                        image = get_image_from_maniskill2_obs_dict(env_unwrapped, obs)
                        image = image[:,:,:3] ## Remove last dimension of image color
                        action, loss = model.forward(torch.tensor(np.array([encode_state(resize_state(image))])).to(device)
                                            ,torch.tensor(np.array([encode_txt(instruction)[:cfg.block_size]])).to(device) ## There can be issues here if th text is shorter than any example in the dataset
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
                
                episode_stats = info.get('episode_stats', {})
                print("Episode stats", episode_stats)
                print(f"avg reward {np.mean(rewards):.8f}")
                if not cfg.testing:
                    wandb.log({"avg reward": np.mean(rewards)})
                import moviepy.editor as mpy
                clip = mpy.ImageSequenceClip(list(frames), fps=20)
                clip.write_videofile(log_dir+"/sim-env-"+str(iter)+".mp4", fps=20)
                if not cfg.testing:
                    wandb.log({"example": wandb.Video(log_dir+"/sim-env-"+str(iter)+".mp4")})

        # sample a batch of data
        xb, xg, xgi, yb = get_batch_grp('train', dataset_tmp, cfg.batch_size)

        # evaluate the loss
        logits, loss = model(xb, xg, xgi, yb)
        loss.backward()

        if (iter + 1) % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if not cfg.testing:
        wandb.finish()
    return losses['val']

if __name__ == "__main__":
    results = my_main()
    print("results:", results)
