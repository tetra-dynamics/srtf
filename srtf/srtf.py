import bisect
from dataclasses import asdict, dataclass
import json
import pathlib
import subprocess
import tempfile
from typing import Dict, List, Literal, Optional, Tuple
import uuid

import torch

TIMESCALE = 600 # evenly divisable by 15, 30, 50, and 60

@dataclass
class EpisodeMetadata:
    name: str
    task: str
    operator: str

    state_size: int
    action_size: int
    num_samples: int
    camera_names: List[str]
    framerate: int
    image_size: Tuple[int, int] # The size all images

@dataclass
class Camera:
    name: str
    src_path: pathlib.Path # The video file containing the camera stream
    src_crop: Optional[Tuple[int, int, int, int]] = None # How to crop from the source stream before resizing
    src_rotation: Optional[Literal['cw', 'ccw']] = None # Whether to optionally rotate the image after cropping

class SRTF:
    def __init__(self, root_dir: pathlib.Path):
        self.root_dir = root_dir

    def all_episode_names(self) -> List[str]:
        return [p.name for p in self.root_dir.iterdir() if p.is_dir() and not p.name.startswith('tmp-') and p.joinpath('metadata.json').exists()]

    def read_metadata(self, episode_name: str) -> EpisodeMetadata:
        with open(self.root_dir.joinpath(episode_name, 'metadata.json'), 'r') as f:
            metadata_obj = json.load(f)
            metadata = EpisodeMetadata(**metadata_obj)
            metadata.image_size = (metadata.image_size[0], metadata.image_size[1])
            return metadata

    def read_samples(self, metadata: EpisodeMetadata, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row_width = metadata.state_size + metadata.action_size
        row_bytes = row_width * 4

        with open(self.root_dir.joinpath(metadata.name, 'states_actions.bin'), 'rb') as f:
            f.seek(start * row_bytes)
            raw = f.read((end - start) * row_bytes)
            if len(raw) != (end - start) * row_bytes:
                raise RuntimeError('unable to read all samples')

        t = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(-1, row_width)
        states = t[:, :metadata.state_size]
        actions = t[:, metadata.state_size:]

        return states, actions

    def read_images(self, metadata: EpisodeMetadata, frame_idx: int) -> torch.Tensor:
        from torchcodec.decoders import VideoDecoder # don't make torchcodec a requirement for generating the dataset

        framerate = metadata.framerate
        ticks_per_frame = TIMESCALE // framerate
        frames = [
            {
                "pts": ticks_per_frame * i,
                "duration": ticks_per_frame,
                "key_frame": 1 if i % framerate == 0 else 0
            }
            for i in range(metadata.num_samples)
        ]
        mapping = json.dumps({"frames": frames})
        video_path = self.root_dir.joinpath(metadata.name, 'combined.mp4')
        decoder = VideoDecoder(str(video_path), custom_frame_mappings=mapping, dimension_order='NHWC')
        frame = decoder.get_frame_at(frame_idx).data  # (n_cams * H, W, C) uint8

        n_cams = len(metadata.camera_names)
        total_H, W, C = frame.shape
        H = total_H // n_cams
        images = frame.reshape(n_cams, H, W, C)
        return images

    def save_episode(self, name: str, states: torch.Tensor, actions: torch.Tensor, cameras: List[Camera],
            task: str = '', operator: str = '', framerate: int = 30, target_image_size: Tuple[int, int] = (224, 224)):
        assert len(states) == len(actions)
        assert len(states.shape) == 2
        assert len(actions.shape) == 2
        assert len(target_image_size) == 2

        num_samples, state_size = states.shape
        action_size = actions.shape[1]

        camera_names = [camera.name for camera in cameras]
        metadata = EpisodeMetadata(name, task, operator, state_size, action_size, num_samples, camera_names, framerate, target_image_size)

        # create it as a tmp dir until the episode is successfully saved, but do it in the root_dir to guarantee atomic rename
        episode_dir = self.root_dir.joinpath(f'tmp-{uuid.uuid4().hex}-{name}')
        episode_dir.mkdir()

        with open(episode_dir.joinpath('metadata.json'), 'w') as f:
            metadata_obj = asdict(metadata)
            json.dump(metadata_obj, f)

        torch.concat([states, actions], dim=1).float().numpy().tofile(episode_dir.joinpath('states_actions.bin'))

        dest_path = episode_dir.joinpath('combined.mp4')
        generate_combined_video(cameras, dest_path, len(states), framerate, target_image_size)

        episode_dir.rename(self.root_dir.joinpath(name))

class EpisodeDataset(torch.utils.data.Dataset):
    '''Simple dataset that provides all full datapoints'''

    def __init__(self, srtf: SRTF, chunk_size: int, episode_names: Optional[List[str]] = None, prefix_exclude_count: int = 0, include_images: bool = True):
        self.srtf = srtf
        self.chunk_size = chunk_size
        # prefix_exclude_count allows you to exclude the first n frames from all episodes
        self.prefix_exclude_count = prefix_exclude_count
        self.include_images = include_images

        self.all_metadata = []
        self.episode_offsets = []
        self.total_datapoints = 0

        if episode_names is None:
            episode_names = srtf.all_episode_names()

        for episode_name in episode_names:
            metadata = srtf.read_metadata(episode_name)
            num_datapoints = metadata.num_samples - (chunk_size - 1) - self.prefix_exclude_count
            if num_datapoints > 0:
                self.all_metadata.append(metadata)
                self.total_datapoints += num_datapoints
                self.episode_offsets.append(self.total_datapoints)

    def __len__(self):
        return self.total_datapoints

    def get_sample_and_metadata(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, EpisodeMetadata, int]:
        '''Helper method if you're wrapping EpisodeDataset with your own logic'''
        episode_idx = bisect.bisect_right(self.episode_offsets, idx)
        metadata = self.all_metadata[episode_idx]

        frame_idx = idx + self.prefix_exclude_count
        if episode_idx > 0:
            frame_idx -= self.episode_offsets[episode_idx - 1]
        states, actions = self.srtf.read_samples(metadata, frame_idx, frame_idx + self.chunk_size)
        state = states[0]

        images = self.srtf.read_images(metadata, frame_idx) if self.include_images else None

        return state, images, actions, metadata, frame_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        state, images, actions, _, _ = self.get_sample_and_metadata(idx)
        return state, images, actions

X264 = ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-bf", "0", "-pix_fmt", "yuv420p"]

def X264_STRICT_FFMPEG_ARGS(fps):
    X264_STRICT_PARAMS = (
        f"keyint={fps}:min-keyint={fps}:scenecut=0:"
        f"fps={fps}/1:timebase=1/{TIMESCALE}:force-cfr=1"
    )
    return [
        "-vsync", "0",
        "-enc_time_base", f"1/{TIMESCALE}",
        "-video_track_timescale", str(TIMESCALE),
        "-bf", "0",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-x264-params", X264_STRICT_PARAMS,
        "-threads", "1",
    ]

def generate_combined_video(cameras: List[Camera], dest: pathlib.Path, num_frames: int, framerate: int, target_image_size: Tuple[int, int]):
    path_to_cameras: Dict[pathlib.Path, List[Camera]] = {}
    for camera in cameras:
        path = camera.src_path
        if path not in path_to_cameras:
            path_to_cameras[path] = []
        path_to_cameras[path].append(camera)

        (video_frames,) = probe(camera.src_path, "-count_frames", "-show_entries", "stream=nb_read_frames")
        if video_frames != num_frames:
            raise ValueError(f'number of video frames ({video_frames}) for video {camera.src_path} must match number of state-action frames ({num_frames})')

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = pathlib.Path(tmpdir_str)

        for path in path_to_cameras:
            generate_downsized_videos(path, path_to_cameras[path], tmpdir, num_frames, framerate, target_image_size)

        mp4s = [tmpdir.joinpath(camera.name + '.mp4') for camera in cameras]

        ticks_per_frame = TIMESCALE // framerate

        # vstack inputs, then force the timebase + integer PTS the trainer expects.
        # settb/setpts make every frame's pts exactly TICKS_PER_FRAME * k in
        # timebase 1/TIMESCALE (matches the synthesized custom_frame_mappings).
        filt = (
            "".join(f"[{i}:v]" for i in range(len(mp4s)))
            + f"vstack=inputs={len(mp4s)}[v0];"
            + f"[v0]settb=expr=1/{TIMESCALE},setpts=N*{ticks_per_frame}[out]"
        )
        subprocess.run(
            ["ffmpeg", "-y", *sum((["-i", str(p)] for p in mp4s), []),
            "-filter_complex", filt, "-map", "[out]",
            *X264_STRICT_FFMPEG_ARGS(framerate), str(dest)],
            capture_output=True, check=True,
        )

def probe(path, *entries):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", *entries, "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip()
    return [int(x) for x in out.split(",")]

def generate_downsized_videos(path: pathlib.Path, cameras: List[Camera], parent_dir: pathlib.Path, num_frames: int, framerate: int, target_size: Tuple[int, int]):
    src_width, src_height = probe(path, "-show_entries", "stream=width,height")

    frame_bytes = src_width * src_height * 3
    decoder = subprocess.Popen(
        ["ffmpeg", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "error", "pipe:1"],
        stdout=subprocess.PIPE,
    )
    assert decoder.stdout

    vf = f'scale={target_size[0]}:{target_size[1]}:force_original_aspect_ratio=decrease:flags=bicubic,pad={target_size[0]}:{target_size[1]}:(ow-iw)/2:(oh-ih)/2'
    encoded_paths = []
    encoders = []
    encoder_inputs = []
    for camera in cameras:
        encoded_path = parent_dir.joinpath(camera.name + '.mp4')
        encoded_paths.append(encoded_path)

        src_crop = camera.src_crop
        if src_crop is None:
            src_crop = center_crop((src_width, src_height), target_size)
        _, _, input_width, input_height = src_crop

        encoder = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{input_width}x{input_height}",
            "-r", str(framerate), "-i", "-", "-vsync", "0", "-vf", vf, *X264, "-threads", "1", encoded_path],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        encoders.append(encoder)
        assert encoder.stdin
        encoder_inputs.append(encoder.stdin)

    try:
        for _ in range(num_frames):
            raw_frame = decoder.stdout.read(frame_bytes)
            if raw_frame is None or len(raw_frame) < frame_bytes:
                raise RuntimeError('not enough frames')

            frame = torch.frombuffer(bytearray(raw_frame), dtype=torch.uint8).reshape(src_height, src_width, 3)
            for camera, encoder_input in zip(cameras, encoder_inputs):
                cropped_frame = frame
                if camera.src_crop:
                    x, y, width, height = camera.src_crop
                    cropped_frame = frame[y:y+height, x:x+width].contiguous()

                if camera.src_rotation is not None:
                    k = -1 if camera.src_rotation == 'cw' else 1
                    cropped_frame = torch.rot90(cropped_frame, k=k, dims=(0, 1))

                cropped_frame_bytes = cropped_frame.numpy().tobytes()
                encoder_input.write(cropped_frame_bytes)
    finally:
        decoder.stdout.close()
        decoder.terminate()
        decoder.wait()
        for encoder_input in encoder_inputs:
            encoder_input.close()
        for encoder in encoders:
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg encode failed")

def center_crop(src_size: Tuple[int, int], target_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    src_aspect = src_size[0] / src_size[1]
    target_aspect = target_size[0] / target_size[1]

    if src_aspect > target_aspect:
        # src is wider than target, crop width
        height = src_size[1]
        width = round(src_size[1] * target_aspect)

        x = (src_size[0] - width) // 2
        y = 0
    elif src_aspect < target_aspect:
        # src is taller than target, crop height
        width = src_size[0]
        height = round(src_size[0] / target_aspect)

        x = 0
        y = (src_size[1] - height) // 2
    else:
        # same aspect ratio
        x = 0
        y = 0
        width = src_size[0]
        height = src_size[1]

    return (x, y, width, height)
