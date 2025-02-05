## Code to fetch data and create an easy dataset.


import hydra, json
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="./conf", config_name="dataset")
def my_main(cfg: DictConfig):
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
    builder = tfds.builder_from_directory(builder_dir=cfg.dataset.from_name)
    datasetRemote = builder.as_dataset(split='train[:' + str(cfg.dataset.num_episodes) + ']')
    dataset_tmp = {"img": [], "action": [], "goal": [], "goal_img": [],
                    "rotation_delta": [], "open_gripper": [] }
    for episode in datasetRemote:
        episode_ = {'steps': [] }
        episode = list(episode['steps'])
        goal_img = cv2.resize(np.array(episode[-1]['observation']['image'], dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  
        for i in range(len(episode)): ## Resize images to reduce computation
            # action = torch.as_tensor(action) # grab first dimention
            obs = cv2.resize(np.array(episode[i]['observation']['image'], dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))
            dataset_tmp["img"].append(Image.fromarray(obs.astype('uint8') ))
            dataset_tmp["action"].append(episode[i]['action']['world_vector'])
            dataset_tmp["rotation_delta"].append(episode[i]['action']['rotation_delta'])
            dataset_tmp["open_gripper"].append([np.array(episode[i]['action']['open_gripper'], dtype=np.uint8)])
            dataset_tmp["goal"].append(episode[i]['observation']['natural_language_instruction'].numpy().decode())
            dataset_tmp["goal_img"].append(Image.fromarray(goal_img.astype('uint8') ))

    print("Dataset shape:", len(dataset_tmp["img"]))
    dataset = {}
    dataset["img"] = dataset_tmp["img"]
    dataset["action"] = np.array(dataset_tmp["action"], dtype=np.float32)
    dataset["rotation_delta"] = np.array(dataset_tmp["rotation_delta"], dtype=np.float32)
    dataset["open_gripper"] = np.array(dataset_tmp["open_gripper"], dtype=np.uint8)
    dataset["goal"] = dataset_tmp["goal"]
    dataset["goal_img"] = dataset_tmp["goal_img"]

    ## Prepare dataset for push to huggingface
    from datasets import Dataset
    import datasets
    from datasets import Image

    ds = Dataset.from_dict(dataset)

    new_features = ds.features.copy()
    new_features["img"] = Image()
    ds.cast(new_features)
    print('Features:', ds.features)
    ds.save_to_disk("datasets/" + cfg.dataset.to_name + ".hf")
    ds.push_to_hub(cfg.dataset.to_name)


if __name__ == "__main__":
    results = my_main()
    print("results:", results)