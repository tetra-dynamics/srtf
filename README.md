# Simple Robot Training Format (SRTF)

This is an implementation of the file format described in the [ABC](https://abc.bot) behavior cloning paper that has been extracted from the [official repo](https://github.com/amazon-far/abc) and refactored slightly in order to make it reusable as a library.

The purpose of SRTF is to be a simple format for robot datasets for training vision-action models. It's not meant to be the source of truth for your robot data, but a format that's optimized for training performance while still being easy to understand. It acheives this in the following ways:

1. All data is stored by episode
2. State and action data is stored in a row oriented flat file so the state and actions for a given sample can be read with a single read
3. All camera images are stored together in a single MP4 file at their resized resolution. That file has the header stored at the beginning, and uses keyframes every second and no B frames such that decoding a random frame of video can be done in a consistent amount of time.

This lets you train over very large datasets of robot data performantly without having to preshuffle data.

## Usage

To use this library, first create a SRTF object. The only parameter is the directory where the SRTF episodes are stored.

```
from srtf import SRTF

srtf = SRTF(root_dir)
```

To populate the directory with episodes, you can use the `save_episode` method, which takes in the states, actions, and information about the cameras and saves them to their respective files.

```
from srtf import Camera

raw_episode_dir = raw_episodes_root.joinpath(episode_name)
cameras = [
    Camera('front', raw_episode_dir.joinpath('front.mp4)),
    Camera('right-palm', raw_episode_dir.joinpath('right-palm.mp4)),
]
srtf.save_episode(episode_name, states, actions, cameras)
```

Images will automatically be resized to 224x224 by default, but that can be changed using the `target_image_size` parameter to the `save_episode` method. By default, images will be center cropped to match the aspect ratio of `target_image_size`, but that can be overridden with the `src_crop` field of the `Camera` object.

The SRTF object provides simple methods to read all the data that has been stored, including `read_metadata`, `read_samples`, and `read_images`.

For simple use cases, there's a `EpisodeDataset` wrapper that creates a PyTorch dataset from all the data in the root directory.

```
dataset = EpisodeDataset(srtf, chunk_size=30)

# pick a sample at random
state, images, actions = dataset[random.randint(0, len(dataset))]
```

You can also pass in a list of episode names using the `episode_names` parameter when creating a dataset if you want to train on a subset of the episodes.

## Installation

There are a couple of ways to install `srtf` right now:

1. Clone the repo and `pip install` it

2. Since the entire implementation is a single file, copy the `srtf.py` file directly into your project

I suspect most people who need something like this will also need to edit it in some way to fit their needs, I'm not putting it on PyPI at the moment, but if people find it useful as is I will reconsider.
