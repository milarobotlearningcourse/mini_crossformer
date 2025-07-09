## Code to fetch data and create an easy dataset.

import hydra, json
from omegaconf import DictConfig, OmegaConf
from transformers import T5Tokenizer, T5ForConditionalGeneration
## import python garbage collector
import gc
gc.enable()
import numpy as np
import torch
import cv2

# from openvla.prismatic.vla.datasets.rlds.oxe import transforms as transforms

def bridge_oxe_dataset_transform(trajectory):
    """
    Applies to version of Bridge V2 in Open X-Embodiment mixture.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    trajectory = trajectory[1:]  # Remove the first timestep with all-zero action

    for i in range(0, len(trajectory)):
        trajectory[i]["action"] = np.concatenate((trajectory[i]['action']['world_vector'], 
                                                    trajectory[i]['action']['rotation_delta'], 
                                                        [trajectory[i]['action']['open_gripper']], 
                                                        ), axis=-1
                                                    ).astype(np.float32),
        trajectory[i]["language_instruction"] = trajectory[i]["observation"]["natural_language_instruction"]
        # trajectory = relabel_bridge_actions(trajectory)
        trajectory[i]["observation"]["EEF_state"] = trajectory[i]["observation"]["state"][:6]
        trajectory[i]["observation"]["gripper_state"] = trajectory[i]["observation"]["state"][-1:]
    return trajectory

def maniskill_dataset_transform(trajectory):
    for i in range(0, len(trajectory)):
        trajectory[i]["observation"]["gripper_state"] = trajectory[i]["observation"]["state"][7:8]
        trajectory[i]["action"] = trajectory[i]["action"].numpy()
        trajectory[i]['observation']["natural_language_instruction"] = trajectory[i]["language_instruction"]
    return trajectory

def robocook_dataset_transform(trajectory):
    for i in range(0, len(trajectory)):
        trajectory[i]["observation"]["eef_state"] = trajectory[i]["observation"]["state"][:6]
        trajectory[i]["action"] = trajectory[i]["action"].numpy()
        trajectory[i]['observation']["natural_language_instruction"] = trajectory[i]["language_instruction"]
        trajectory[i]["observation"]["gripper_state"] = trajectory[i]["observation"]["state"][-1:]
        trajectory[i]["observation"]["image"] = trajectory[i]["observation"]["image_1"]
    return trajectory

def apply_transforms(episode, cfg, dataset_name):
    """
    Apply the necessary transforms to the episode data.
    This function is a placeholder for any transformations that need to be applied.
    """
    TRANSFORMS = {
        "bridge_oxe": bridge_oxe_dataset_transform,
        "stanford_robocook_converted_externally_to_rlds": robocook_dataset_transform,
        "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_transform,
        # Add other dataset specific transforms here if needed
    }
    # Example transformation: resize images, normalize actions, etc.
    episode = TRANSFORMS[cfg.dataset.dataset_indicies[dataset_name]["dataset_key"]](episode)
    return episode

class CircularBuffer:
    """ A circular buffer impolimented using a collection of numpy arrays.
    The buffer stores images, actions, goals, goal images, rotation deltas, and open gripper states.
    The buffer has a fixed size and overwrites old data when full.
    The buffer is initialized with a size and a configuration object.
    """
    def __init__(self, size, cfg):
        import tensorflow_datasets as tfds
        self._size = size
        self._cfg = cfg
        self._index = 0
        self._count = 0
        self._dataset_tmp = {
                            "img": torch.tensor(np.zeros(shape=(self._size, self._cfg.image_shape[0], self._cfg.image_shape[0], 3)), dtype=torch.uint8, device=self._cfg.device), 
                            "action": torch.tensor(np.zeros(shape=(self._size, len(self._cfg.env.action_std)),), dtype=torch.float, device=self._cfg.device),
                            "goal": torch.tensor(np.zeros(shape=(self._size, self._cfg.max_block_size)), dtype=torch.float, device=self._cfg.device), 
                            "goal_img": torch.tensor(np.zeros(shape=(self._size, self._cfg.image_shape[0], self._cfg.image_shape[0], 3)), dtype=torch.uint8, device=self._cfg.device),
                            # "rotation_delta": [], "open_gripper": [] 
                            "t5_language_embedding": torch.tensor(np.zeros(shape=(self._size, 1, self._cfg.n_embd)), dtype=torch.float, device=self._cfg.device) if self._cfg.dataset.encode_with_t5 else None,
                            "terminal": torch.tensor(np.zeros(shape=(self._size, 1)), dtype=torch.uint8, device=self._cfg.device),
                            } 
                    
        if self._cfg.dataset.encode_with_t5:
            self._tokenizer = T5Tokenizer.from_pretrained(self._cfg.dataset.t5_version)
            self._model = T5ForConditionalGeneration.from_pretrained(self._cfg.dataset.t5_version)
            # self._dataset_tmp["t5_language_embedding"] = torch.tensor(np.zeros(shape=(self._size, self._cfg.max_block_size, self._cfg.n_embd)), dtype=torch.float, device=self._cfg.device)[0],  

        self._builders = {}
        for dataset_name in self._cfg.dataset.dataset_indicies:
            self._builders[dataset_name] = tfds.builder_from_directory(builder_dir=dataset_name)
        ## Get the size of the dataset from the builder
        # info = self._builder
        # print(f"Total number of examples (from loaded info): {info.splits.total_num_examples}")
        # self._max_size = info.splits.total_num_examples

        chars = cfg.dataset.chars_list
        print("chars", chars)
        cfg.vocab_size = len(chars)
        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        self._encode_txt = lambda s: [stoi[c] for c in s] # text encoder to tokens: 
        self._decode_txy = lambda l: ''.join([itos[i] for i in l]) # token decoder to text: 
        print("vocab_size:", cfg.vocab_size)

            ## Get the actions and encode them to map to [-1, 1]
        self._encode_state = lambda af:   ((af/(255.0)*2.0)-1.0) # encoder: take a float, output an integer
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        # print("example text encode:", encode_txt(dataset_tmp["goal"][0]))

        cfg.action_bins = len(cfg.env.action_mean)
        self._encode_action = lambda af:   (((af - cfg.env.action_mean)/(cfg.env.action_std))).astype(np.float32) # encoder: take a float, output an integer
        self._decode_action = lambda binN: (binN * cfg.env.action_std) + cfg.env.action_mean  # Undo mapping to [-1, 1]

        if self._cfg.dataset.load_dataset:
            # Load the dataset from a file
            import datasets
            dataset = datasets.load_dataset(self._cfg.dataset.to_name, split='train')
            dataset_tmp = {
                "img": np.array(dataset["img"]),
                "action": np.concatenate((np.array(dataset["action"]) ,np.array(dataset["rotation_delta"])
                                        ,np.array(dataset["open_gripper"])), axis=1),
                "goal_img": np.array(dataset["goal_img"]),
                "goal": dataset["goal"]
            }
            for i in range(len(dataset_tmp["img"])):
                if len(dataset_tmp["action"][i:i+self._cfg.policy.action_stacking]) < self._cfg.policy.action_stacking:
                    print("Skipping index", i, "because action length is less than", self._cfg.policy.action_stacking)
                    continue
                self.add(
                        dataset_tmp["img"][i], 
                        #  np.reshape(dataset_tmp["img"][i:i+self._cfg.policy.action_stacking], newshape=(1, len(self._cfg.env.action_std) * self._cfg.policy.action_stacking) ),
                          dataset_tmp["action"][i],
                          dataset_tmp["goal"][i], 
                          dataset_tmp["goal_img"][i],
                        #   language_instruction=dataset["language_instruction"][i] if cfg.dataset.encode_with_t5 else None
                        terminal=0
                          )
                # self.add(dataset_tmp["img"][i], , goal, goal_img, language_instruction)
            print("Loaded dataset with size:", self._count)
        self._dataset_indecies = self._cfg.dataset.dataset_indicies

    def add(self, obs, action, goal, goal_img, language_instruction=None, terminal=0):
        """ Add an observation, action, goal, goal image, rotation delta, and open gripper state to the buffer."""
    
        self._dataset_tmp["img"][self._index] = torch.tensor(obs, dtype=torch.uint8, device=self._cfg.device)
        self._dataset_tmp["action"][self._index] = torch.tensor(action, dtype=torch.float, device=self._cfg.device)
        ## Make goal embeddings of a fixed length and fill in the earlier chunks with the true goal data
        
        if self._cfg.dataset.encode_with_t5:
            goal__ = np.zeros((self._cfg.n_embd))
            input_ids = self._tokenizer(goal, return_tensors="pt").input_ids
            goal_t = self._model.encoder(input_ids).last_hidden_state.detach().cpu().numpy()[0, -1] ## Get the goal embedding
            # goal__[:len(goal_t[0]), :] = goal_t[0][:self._cfg.max_block_size] ## Overwrite just the zeros up to the size of this vector, smaller vectors will have < max_block_size
            self._dataset_tmp["t5_language_embedding"][self._index] = torch.tensor(goal_t, dtype=torch.float, device=self._cfg.device)
        
        goal_ = " " * self._cfg.max_block_size
        goal_ = goal[:self._cfg.max_block_size] + goal_[len(goal):self._cfg.max_block_size] 
        # assert len(goal_) == self._cfg.max_block_size
        self._dataset_tmp["goal"][self._index] = torch.tensor(self._encode_txt(goal_), dtype=torch.float, device=self._cfg.device)
        self._dataset_tmp["goal_img"][self._index] = torch.tensor(goal_img, dtype=torch.uint8, device=self._cfg.device)
        self._dataset_tmp["terminal"][self._index] = torch.tensor(terminal, dtype=torch.uint8, device=self._cfg.device)
        self._count += 1
        self._index = (self._index + 1) % self._size

    def get_batch_grp(self, split, cfg, batch_size):
        # generate a small batch of inputs x and targets y
        # data = dataset['train'] if split == 'train' else dataset['test']
        data = self._dataset_tmp
        ix = np.random.randint(min(self._count, self._size)-(max(cfg.policy.action_stacking, cfg.policy.obs_stacking)-1), size=(batch_size,))
        if cfg.policy.obs_stacking > 1:
            obs_ = torch.concatenate((data["img"][ix], data["img"][ix+1]), axis=-1) 
            x = torch.tensor(self._encode_state(obs_), dtype=torch.float, device=cfg.device)
        else:
            x = torch.tensor(self._encode_state(data["img"][ix]), dtype=torch.float, device=cfg.device)
        if cfg.dataset.encode_with_t5:
            x_goal = torch.tensor(data["t5_language_embedding"][ix], dtype=torch.float, device=cfg.device)
        else:
            x_goal = torch.tensor(data["goal"][ix], dtype=torch.long, device=cfg.device)
        x_goal_img = torch.tensor(self._encode_state(data["goal_img"][ix]), dtype=torch.float, device=cfg.device)
        if cfg.policy.action_stacking > 1:
            ## Stack the next cfg.policy.action_stacking actions together
            ## Can extended slicing us list of lists...
            y = torch.concatenate((data["action"][ix], data["action"][ix+1]), axis=1) 
        else:
            y = torch.tensor(data["action"][ix], dtype=torch.float, device=cfg.device)

        return x, x_goal, x_goal_img, y
    
    def shuffle(self, shared_queue):
        print("num", shared_queue)
        while True:
            data = shared_queue.get() ## Update the data when messaged from the Queue
            if data is None:
                break
            start_ = self._dataset_indecies[self._cfg.dataset.from_name]["start"]
            ## Call function to swap out a portion of data.
            get_multi_dataset_portion(self._builders, self, self._cfg)

    def save(self, path):
        """
        Save the dataset to a file.
        """
        ## Prepare dataset for push to huggingface
        from datasets import Dataset
        import datasets
        from datasets import Image

        ds = Dataset.from_dict(self._dataset_tmp)

        new_features = ds.features.copy()
        new_features["img"] = Image()
        ds.cast(new_features)
        print('Features:', ds.features)
        # ds.save_to_disk("datasets/" + cfg.dataset.to_name + ".hf")
        ds.push_to_hub(self._cfg.dataset.to_name)

def get_dataset_portion(builder, cbuffer, start, end, cfg, dataset_name=None):
    """
    Helper function to get a portion of the dataset.
    """
    import tensorflow_datasets as tfds
    import numpy as np
    from tqdm import tqdm, trange
    import cv2
    from PIL import Image
    from datasets import load_dataset
    # ------------
    # Train and test splits
    # Loading data
    # create RLDS dataset builder
    for c in range(start, end, cfg.dataset.chunk_size):
        datasetRemote = builder.as_dataset(split='train[' + str(c) + ':' + str(c + cfg.dataset.chunk_size) + ']')
        # print("loading dataset chunk:", c, "to", c + cfg.dataset.chunk_size)
        gc.collect()
        for episode in datasetRemote:
            episode = list(episode['steps'])
            ## https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/transforms.py
            episode = apply_transforms(episode, cfg, dataset_name)
            goal_img = cv2.resize(np.array(episode[-1]['observation']["image"], dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  
            # print("Ajout de", len(episode), "données à la circular buffer.")
            for i in range(len(episode)): ## Resize images to reduce computation
                if (i+cfg.policy.action_stacking > len(episode)):
                    # print("Skipping index", i, "because action length is less than", cfg.policy.action_stacking)
                    continue
                obs = cv2.resize(np.array(episode[i]['observation']["image"], dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))
                cbuffer.add(obs = obs, 
                            action = episode[i]['action'],
                            goal= episode[i]['observation']["natural_language_instruction"].numpy().decode(),
                            # goal=episode[i]['observation']['natural_language_instruction'],
                            goal_img=goal_img,
                            # rotation_delta=episode[i]['action']['rotation_delta'], 
                            # language_instruction=episode[i]['observation']['natural_language_instruction'].numpy().decode()
                            terminal = 1 if i == len(episode) - 1 else 0
                            )
    print("A terminé le mélange.")
    return cbuffer

def get_multi_dataset_portion(builders, cbuffer, cfg):
    """
    Helper function to get a portion of the dataset.
    """
    import tensorflow_datasets as tfds
    import numpy as np
    from tqdm import tqdm, trange
    import cv2
    # from PIL import Image
    from datasets import load_dataset
    # ------------
    # Train and test splits
    # Loading data
    # create RLDS dataset builder
    for dataset_name, builder in builders.items():
        print("Loading dataset:", dataset_name)
        start = cfg.dataset.dataset_indicies[dataset_name]["start"]
        end = start + int(cfg.dataset.chunk_size * cfg.dataset.dataset_indicies[dataset_name]["weight"])
        print("start:", start, "end:", end, "dataset_name:", dataset_name)
        print("start_", start, " end_", end, " size_ ", cbuffer._dataset_indecies[dataset_name]["size"]
                , " count_", cbuffer._count, " index_", cbuffer._index)
        if start >= cbuffer._dataset_indecies[dataset_name]["size"]: ## If we have reached the end of the dataset, reset the start index
            cbuffer._dataset_indecies[dataset_name] = 0
        else:
            cfg.dataset.dataset_indicies[dataset_name]["start"] = end
        get_dataset_portion(builders[dataset_name], cbuffer, start, end, cfg, dataset_name=dataset_name)

@hydra.main(config_path="./conf", config_name="mix-64")
def my_main(cfg: DictConfig):
    import tensorflow_datasets as tfds
    import numpy as np
    from tqdm import tqdm, trange
    import cv2
    from PIL import Image
    # ------------
    # Train and test splits
    # Loading data
    # create RLDS dataset builder
    cbuffer = CircularBuffer(cfg.dataset.buffer_size, cfg)
    for i in range(0, cfg.dataset.num_episodes, cfg.dataset.chunk_size):
        # get_dataset_portion(builder, cbuffer, 0, cfg.dataset.num_episodes, cfg)
        # cbuffer.shuffle()
        ## Call function to swap out a portion of data.
        get_multi_dataset_portion(cbuffer._builders, cbuffer, cbuffer._cfg)
        print("Dataset shape:", len(cbuffer._dataset_tmp["img"]))
        print("Dataset len:", cbuffer._count)


if __name__ == "__main__":
    results = my_main()
    print("results:", results)