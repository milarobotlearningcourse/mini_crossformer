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


@torch.no_grad()
def estimate_loss(model, dataset):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X, x_goal, x_goal_img, Y = dataset.get_batch_grp(split, model._cfg, model._cfg.batch_size)
            logits, loss = model(X, x_goal, x_goal_img, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_patches_fast(images, cfg):
    from einops import rearrange
    batch_size, height, width, channels = images.shape
    patch_size = cfg.patch_size ## n_patches = 8
    patch_items = 192 ## 192 is the number of items in each patch (8 height * 8 width * 3 channels)

    patches = rearrange(images[:,:,:,:3], 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
    if channels > 3:
        ## History stacking in the channel dimension for observations only, not goal images.
        # patches_2 = rearrange(images[:,:,:,3:], 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        patches = rearrange(images, 'b (h p1) (w p2) (c hs) -> b (h w hs) (p1 p2 c)', p1 = patch_size, p2 = patch_size, hs=cfg.policy.obs_stacking) ## Stack the history in the channel dimension
        # patches = torch.cat((patches, patches_2), dim=1) ## Conc
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
        # [TODO]
        """
        [DEFAULT]
        # TODO: 
        ## Provide the block masking
        pass
        [/DEFAULT]
        """
        if mask == None:
            mask = torch.ones((T, ), device=self.device) ## (1, T)
        # [/TODO]
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
            nn.Linear(n_embd, 4 * n_embd), ## This is where the information may be sotred.
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
  def __init__(self, cfg, mlp_ratio=4):
    super(GRP, self).__init__()
    # self._dataset = dataset
    self._cfg = cfg
    # [TODO]
    """
    [DEFAULT]
    # TODO: 
    ## Provide the logic for the GRP network

    # 4) Transformer encoder blocks

    # 5) Classification MLPk
    
    [/DEFAULT]
    """
    self.patch_size = (self._cfg.image_shape[0] / self._cfg.n_patches, self._cfg.image_shape[1] / self._cfg.n_patches)
    #Positional embedding
    self.register_buffer('positional_embeddings', calc_positional_embeddings(1 + ((self._cfg.n_patches ** 2) * self._cfg.policy.obs_stacking) + self._cfg.max_block_size + self._cfg.n_patches ** 2, cfg.n_embd), persistent=False)

    self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
    self.class_tokens = nn.Parameter(torch.rand(1, cfg.n_embd))

    self.input_d = int(self._cfg.image_shape[2] * self.patch_size[0] * self.patch_size[1])

    self.lin_map = nn.Linear(self.input_d, self._cfg.n_embd, bias=False) 
    self.lin_map_pose = nn.Linear(7, self._cfg.n_embd, bias=True) 

    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([Block(self._cfg.n_embd, self._cfg.n_head, dropout=self._cfg.dropout) for _ in range(self._cfg.n_blocks)])

    # 5) Classification MLPk
    self.mlp = nn.Sequential(
        nn.Linear(self._cfg.n_embd, self._cfg.action_bins * self._cfg.policy.action_stacking),  # Output size is action_bins * action_stacking
    )
    # [/TODO]

  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, images, goals_txt, goal_imgs, targets=None, pose=None):
    # Dividing images into patches
    n, c, h, w = images.shape
    # [TODO]
    """
    [DEFAULT]
    # TODO: 
    ## Provide the logic to produce the output and loss for the GRP
    
    # Map the vector corresponding to each patch to the hidden size dimension

    # Adding classification and goal_img tokens to the tokens

    # Adding positional embedding

    # Compute blocked masks

    # Transformer Blocks

    # Getting the classification token only

    # Compute output and loss

    [/DEFAULT]
    """
    # patches = get_patches_fast(images[:,:,:,:3]) ## Only use the first 3 channels of the image
    # patches_more = get_patches_fast(images[:,:,:,3:])
    # obs_patches = [get_patches_fast(images[:,:,:,3*i:3*(i+1)] for i in range(self._cfg.policy.obs_stacking))] ## Only use the first 3 channels of the image
    obs_patches = get_patches_fast(images, self._cfg) 
    patches_g = get_patches_fast(goal_imgs, self._cfg)
    if self._cfg.dataset.encode_with_t5:
        goals_e = goals_txt ## This is actually the embedding from the T5 model
        B, T, E = goals_txt.shape
        # T = 1
        goals_e = torch.reshape(goals_e, (B, 1, E)) ## Reshape to match the embedding size
    else:
        goals_e = self.token_embedding_table(goals_txt)
        B, E = goals_txt.shape
        T = self._cfg.max_block_size
    
    # Running linear layer tokenization to get embeddings
    # Map the vector corresponding to each patch to the hidden size dimension
    # out = self.lin_map(patches)
    # out_m = self.lin_map(patches_more)
    out_obs = self.lin_map(obs_patches) ## List of tensors, one for each stacked observation
    out_g = self.lin_map(patches_g)
    
    # Adding classification and goal_img embeddings to the other embeddings
    out = torch.cat((self.class_tokens.expand(n, 1, -1), out_obs, goals_e, out_g), dim=1)
    
    # Adding positional embedding
    out = out + self.positional_embeddings.repeat(n, 1, 1)

    ## Compute blocked masks
    mask = torch.ones((1 + (c * self._cfg.policy.obs_stacking) + T + c, ), device=self._cfg.device) ## (1, T)
    if targets is None:
        pass
    elif (torch.rand(1)[0] > 0.66):  
        mask[1 + (c * self._cfg.policy.obs_stacking): 1 + (c * self._cfg.policy.obs_stacking) + T] = torch.zeros((1,T), device=self._cfg.device) ## Mask goal string
    elif (torch.rand(1)[0] > 0.33):
        mask[1 + (c * self._cfg.policy.obs_stacking) + T: 1 + (c * self._cfg.policy.obs_stacking) + T + c] = torch.zeros((1,c), device=self._cfg.device) ## Mask goal image
        
    # Transformer Blocks
    for block in self.blocks:
        out = block(out, mask)

    # Getting the classification token only
    out = out[:, 0]
    out = self.mlp(out)
        
    if targets is None:
        loss = None
    else:
        B, C = out.shape
        loss = F.mse_loss(out, targets) ## B, C
    # [/TODO]
    return (out, loss)
  
from sim_eval import eval_model_in_sim

import hydra, json
from omegaconf import DictConfig, OmegaConf
from mini_shuffel_buffer import CircularBuffer, get_dataset_portion
import threading
from queue import Queue

def preprocess_data(cfg, device):
    from datasets import load_dataset, load_from_disk
    cbuffer = CircularBuffer(cfg.dataset.buffer_size, cfg)

    return cbuffer


# @hydra.main(config_path="conf", config_name="grp-mini")
@hydra.main(config_path="./conf", config_name="libero-64pix")
def my_main(cfg: DictConfig):
    torch.manual_seed(cfg.r_seed)
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print ("cfg:", OmegaConf.to_yaml(cfg))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    cfg.device = device

    wandb = None
    if not cfg.testing:
        import wandb
        # start a new wandb run to track this script
        wandb.init(
            project=cfg.experiment.project,
            # track hyperparameters and run metadata
            config= OmegaConf.to_container(cfg)
        )
        wandb.run.log_code(".")

    tokenizer = None
    text_model = None
    if cfg.dataset.encode_with_t5: ## Load T5 model
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        tokenizer = T5Tokenizer.from_pretrained(cfg.dataset.t5_version)
        text_model = T5ForConditionalGeneration.from_pretrained(cfg.dataset.t5_version)

    cBuffer = preprocess_data(cfg, device)
    model = GRP(cfg)
    model.to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    ## Print the amount of memory used by the model
    print("Memory used by the model:", torch.cuda.memory_allocated(device) / 1e6, "MB")
    ## Print the amount of memory used by the dataset cBuffer
    from pympler import asizeof
    print("Memory used by the dataset cBuffer:", asizeof.asizeof(cBuffer) / 1e6, "MB")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    import torch.optim.lr_scheduler as lr_scheduler
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.max_iters)

    if cfg.simEval == "simpler_env":
        import simpler_env
        task_name = "widowx_carrot_on_plate"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
        if 'env' in locals():
            print("Closing existing env")
            env.close()
            del env
        env = simpler_env.make(task_name)
        env_unwrapped = env.env.env.env ## Updated gymnasium wrapper adds lots of wrappers.

    shared_queue = Queue(maxsize=1)
    data_thread = threading.Thread(target=cBuffer.shuffle, args=(shared_queue,))
    data_thread.start()

    for iter in range(cfg.max_iters):

        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
            losses = estimate_loss(model, cBuffer)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, memory {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
            if not cfg.testing:
                wandb.log({"train loss": losses['train'], "val loss": losses['val']})

        if iter % cfg.data_shuffel_interval == 0 or iter == cfg.max_iters - 1:
            torch.save(model, "miniGRP.pth")
            print("Model saved to miniGRP.pth")
        if cfg.simEval and (iter % cfg.eval_vid_iters == 0): ## Do this eval infrequently because it takes a fiar bit of compute
            if cfg.simEval == "libero":
                from sim_eval import eval_libero
                eval_libero(cBuffer, model, device=cfg.device, cfg=cfg, iter_=iter, log_dir=log_dir, 
                            tokenizer=tokenizer, text_model=text_model, wandb=wandb)
            elif cfg.simEval == "simpler_env":
                eval_model_in_sim(cfg, model, device, log_dir, env, env_unwrapped, 
                              cBuffer, wandb=wandb, iter_=iter, tokenizer=tokenizer, text_model=text_model)


        if iter % cfg.data_shuffel_interval == 0 and iter > 0:
            ## Update the dataset
            shared_queue.put('shuffle')

        xb, xg, xgi, yb = cBuffer.get_batch_grp('train', cfg, cfg.batch_size)

        # evaluate the loss
        logits, loss = model(xb, xg, xgi, yb)
        loss.backward()

        if (iter + 1) % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if not cfg.testing:
        wandb.finish()
    shared_queue.put(None)
    data_thread.join()

    return losses['val']
 
if __name__ == "__main__":
    results = my_main()
    print("results:", results)