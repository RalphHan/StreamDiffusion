import os
import sys
from typing import Literal, Dict, Optional

import fire
import numpy as np
import torch
import cv2
from torchvision.io import read_video, write_video
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
        input: str,
        output: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "output.mp4"),
        model_id: str = "KBlueLeaf/kohaku-v2.1",
        lora_dict: Optional[Dict[str, float]] = None,
        prompt: str = "1girl with brown dog ears, thick frame glasses",
        scale: float = 1.0,
        acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
        use_denoising_batch: bool = True,
        enable_similar_image_filter: bool = True,
        seed: int = 2,
):
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    input : str, optional
        The input video name to load images from.
    output : str, optional
        The output video name to save images to.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {"LoRA_1" : 0.5 , "LoRA_2" : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    scale : float, optional
        The scale of the image, by default 1.0.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    enable_similar_image_filter : bool, optional
        Whether to enable similar image filter or not,
        by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    """
    cap = cv2.VideoCapture(input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id,
        lora_dict=lora_dict,
        t_index_list=[35, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=False,
        mode="img2img",
        output_type="pt",
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=0.98,
        use_denoising_batch=use_denoising_batch,
        seed=seed,
    )

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    # video_result = torch.zeros(video.shape[0], height, width, 3)
    frame = torch.from_numpy(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)).permute(2, 0, 1) / 255
    for _ in range(stream.batch_size):
        stream(image=frame)
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = torch.from_numpy(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)).permute(2, 0, 1) / 255
        output_image = cv2.cvtColor(np.clip((stream(frame) * 255).permute(1, 2, 0).numpy(), 0, 255).astype(np.uint8),
                                    cv2.COLOR_RGB2BGR)
        out.write(output_image)
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)

    cap.release()
    out.release()


if __name__ == "__main__":
    fire.Fire(main)
