import os
import pathlib
import tempfile

import torch

from srtf import Camera, EpisodeDataset, SRTF
from srtf.srtf import center_crop

def main():
    test_center_crop()

    video_dir = pathlib.Path('data')

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = pathlib.Path(tmpdir_str)
        srtf = SRTF(tmpdir)

        episode_names = ['1', '2']
        camera_names = ['front', 'right-palm']
        for episode_name in episode_names:
            episode_dir = video_dir.joinpath(episode_name)
            cameras = [Camera(name, episode_dir.joinpath(name + '.mp4')) for name in camera_names]
            for camera in cameras:
                if not camera.src_path.exists():
                    raise Exception(f'data/{episode_name} must contain front.mp4 and right-palm.mp4')

            states = torch.tensor([[0], [1], [2], [3], [4]], dtype=torch.float32) * int(episode_name)
            actions = torch.tensor([[0.5], [1.5], [2.5], [3.5], [4.5]], dtype=torch.float32) * int(episode_name)

            srtf.save_episode(episode_name, states, actions, cameras)

        dataset = EpisodeDataset(srtf, 2)

        assert len(dataset) == 8
        all_states = []
        all_actions = []
        for i in range(len(dataset)):
            state, images, actions = dataset[i]
            all_states.append(state)
            all_actions.append(actions)
            assert images.shape == (2, 224, 224, 3)

        state_batch = torch.stack(all_states)
        actions_batch = torch.stack(all_actions)

        state_batch_expected = torch.tensor([[0], [1], [2], [3], [0], [2], [4], [6]], dtype=torch.float32)
        actions_expected = torch.tensor([
            [[0.5], [1.5]],
            [[1.5], [2.5]],
            [[2.5], [3.5]],
            [[3.5], [4.5]],
            [[1], [3]],
            [[3], [5]],
            [[5], [7]],
            [[7], [9]],
        ])

        assert (state_batch == state_batch_expected).all()
        assert (actions_batch == actions_expected).all()

def test_center_crop():
    assert center_crop((640, 480), (224, 224)) == (80, 0, 480, 480)
    assert center_crop((640, 480), (480, 360)) == (0, 0, 640, 480)

if __name__ == '__main__':
    main()