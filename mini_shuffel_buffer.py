## Code to fetch data and create an easy dataset.

import hydra, json
from omegaconf import DictConfig, OmegaConf
from transformers import T5Tokenizer, T5ForConditionalGeneration
## import python garbage collector
import gc
gc.enable()
import numpy as np
import torch

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
        self._dataset_tmp = {"img": np.zeros(shape=(self._size, self._cfg.image_shape[0], self._cfg.image_shape[0], 3)), 
                             "action": np.zeros(shape=(self._size, len(self._cfg.env.action_std)),),
                             "goal": np.zeros(shape=(self._size, self._cfg.max_block_size)), 
                             "goal_img": np.zeros(shape=(self._size, self._cfg.image_shape[0], self._cfg.image_shape[0], 3)),
                    # "rotation_delta": [], "open_gripper": [] 
                    }
        if self._cfg.dataset.encode_with_t5:
            self._tokenizer = T5Tokenizer.from_pretrained(self._cfg.dataset.t5_version)
            self._model = T5ForConditionalGeneration.from_pretrained(self._cfg.dataset.t5_version)
            self._dataset_tmp["t5_language_embedding"] = [] 

        self._builder = tfds.builder_from_directory(builder_dir=cfg.dataset.from_name)

        chars = cfg.dataset.chars_list
        print("chars", chars)
        cfg.vocab_size = len(chars)
        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        self._encode_txt = lambda s: [stoi[c] for c in s] # text encoder to tokens: 
        self._decode_txy = lambda l: ''.join([itos[i] for i in l]) # token decoder to text: 
        print("vocab_size:", cfg.vocab_size)
        # print("example text encode:", encode_txt(dataset_tmp["goal"][0]))

        if self._cfg.dataset.load_dataset:
            # Load the dataset from a file
            import datasets
            dataset = datasets.load_dataset(self._cfg.dataset.to_name, split='train')
            dataset_tmp = {
                "img": np.array(dataset["img"]),
                "action": np.concatenate((np.array(dataset["action"]) ,np.array(dataset["rotation_delta"])
                                        ,np.array(dataset["open_gripper"])), axis=1),
                "goal_img": np.array(dataset["goal_img"]),
                "goal": dataset["t5_language_embedding"] if cfg.dataset.encode_with_t5 else dataset["goal"]
            }
            for i in range(len(dataset_tmp["img"])):
                self.add(dataset_tmp["img"][i], 
                          dataset_tmp["action"][i], 
                          dataset_tmp["goal"][i], 
                          dataset_tmp["goal_img"][i],
                          language_instruction=dataset["language_instruction"][i] if cfg.dataset.encode_with_t5 else None)
                # self.add(dataset_tmp["img"][i], , goal, goal_img, language_instruction)
            print("Loaded dataset with size:", self._count)
        self._dataset_indecies = self._cfg.dataset.dataset_indicies

    def add(self, obs, action, goal, goal_img, language_instruction=None):
        """ Add an observation, action, goal, goal image, rotation delta, and open gripper state to the buffer."""
    
        self._dataset_tmp["img"][self._index] = obs
        self._dataset_tmp["action"][self._index] = action
        ## Make goal embeddings of a fixed length and fill in the earlier chunks with the true goal data
        goal_ = np.zeros((self._cfg.max_block_size, self._cfg.n_embd)) if self._cfg.dataset.encode_with_t5 else " " * self._cfg.max_block_size
        if self._cfg.dataset.encode_with_t5:
            goal_[:len(goal[0]), :] = goal[0][:self._cfg.max_block_size] ## Overwrite just the zeros up to the size of this vector, smaller vectors will have < max_block_size
        else:
            goal_ = goal[:self._cfg.max_block_size] + goal_[len(goal):self._cfg.max_block_size] 
            assert len(goal_) == self._cfg.max_block_size
        self._dataset_tmp["goal"][self._index] = self._encode_txt(goal_)
        self._dataset_tmp["goal_img"][self._index] = goal_img
        # self._dataset_tmp["rotation_delta"][self._index] = rotation_delta
        # self._dataset_tmp["open_gripper"][self._index] = open_gripper
        if self._cfg.dataset.encode_with_t5:
            input_ids = self._tokenizer(language_instruction, return_tensors="pt").input_ids
            self._dataset_tmp["t5_language_embedding"][self._index] = self._model.encoder(input_ids).last_hidden_state
        self._count += 1
        self._index = (self._index + 1) % self._size

    def get_batch_grp(self, split, cfg, batch_size):
        # generate a small batch of inputs x and targets y
        # data = dataset['train'] if split == 'train' else dataset['test']
        data = self._dataset_tmp
        ix = np.random.randint(min(self._count, self._size), size=(batch_size,))
        x = torch.tensor(data["img"][ix], dtype=torch.float, device=cfg.device)
        if cfg.dataset.encode_with_t5:
            x_goal = torch.tensor(data["goal"][ix], dtype=torch.float, device=cfg.device)
        else:
            x_goal = torch.tensor(data["goal"][ix], dtype=torch.long, device=cfg.device)
        x_goal_img = torch.tensor(data["goal_img"][ix], dtype=torch.float, device=cfg.device)
        y = torch.tensor(data["action"][ix], dtype=torch.float, device=cfg.device)
        return x, x_goal, x_goal_img, y
    
    def shuffle(self, num):
        print("num", num)
        start_ = self._dataset_indecies["gs://gresearch/robotics/bridge/0.1.0/"]
        get_dataset_portion(self._builder, self, start_, start_ + self._cfg.dataset.chunk_size, self._cfg)
        start_ = self._dataset_indecies["gs://gresearch/robotics/bridge/0.1.0/"] = start_ + self._cfg.dataset.chunk_size

def get_dataset_portion(builder, cbuffer, start, end, cfg):
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
            goal_img = cv2.resize(np.array(episode[-1]['observation']['image'], dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  
            for i in range(len(episode)): ## Resize images to reduce computation
                
                obs = cv2.resize(np.array(episode[i]['observation']['image'], dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))
                cbuffer.add(obs = obs, 
                            action = np.concatenate((episode[i]['action']['world_vector'], 
                                                    episode[i]['action']['rotation_delta'], 
                                                    [np.array(episode[i]['action']['open_gripper'], dtype=np.uint8)]), axis=0).astype(np.float32), 
                            goal= episode[i]['observation']['natural_language_instruction'].numpy().decode(),
                            # goal=episode[i]['observation']['natural_language_instruction'],
                            goal_img=goal_img,
                            # rotation_delta=episode[i]['action']['rotation_delta'], 
                            # language_instruction=episode[i]['observation']['natural_language_instruction'].numpy().decode()
                            )
    print("A terminé le mélange.")

    return cbuffer

@hydra.main(config_path="./conf", config_name="dataset-shuffle")
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
    for i in range(cfg.dataset.starting_episode, cfg.dataset.num_episodes, cfg.dataset.chunk_size):
        # get_dataset_portion(builder, cbuffer, 0, cfg.dataset.num_episodes, cfg)
        cbuffer.shuffle()
        print("Dataset shape:", len(cbuffer._dataset_tmp["img"]))
        print("Dataset len:", len(cbuffer._count))


if __name__ == "__main__":
    results = my_main()
    print("results:", results)