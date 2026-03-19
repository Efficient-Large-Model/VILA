<div align="center">

# NVILA-HD-Video

[![Website](https://img.shields.io/badge/Website-76b900?style=for-the-badge&logo=safari&labelColor=555555)](https://autogaze.github.io/)
[![Arxiv](https://img.shields.io/badge/Arxiv-b31b1b?style=for-the-badge&logo=arxiv&labelColor=555555)](https://arxiv.org/abs/ARXIV_ID)
[![Models & Data & Benchmark](https://img.shields.io/badge/Models%20%26%20Data%20%26%20Benchmark-ffd21e?style=for-the-badge&logo=huggingface&labelColor=555555)](https://huggingface.co/collections/bfshi/autogaze)
[![Demo](https://img.shields.io/badge/Demo-ff6e00?style=for-the-badge&logo=huggingface&labelColor=555555)](https://huggingface.co/spaces/bfshi/AutoGaze)
[![AutoGaze Code](https://img.shields.io/badge/AutoGaze%20Code%20-181717?style=for-the-badge&logo=github&labelColor=555555)](https://github.com/NVlabs/AutoGaze)

</div>

## TL;DR

NVILA-HD-Video is an 8B-parameter multimodal large language model (MLLM) that understands and answers questions about videos with up to **4K resolution** and **1K frames**. It uses [AutoGaze](https://github.com/NVlabs/AutoGaze) (Autoregressive Gazing) to automatically identify and remove redundant patches in a video before running the vision encoder or LLM. Empirically, AutoGaze reduces the number of tokens in a video by up to **100x**, cutting ViT latency by up to **19x** and LLM latency by up to **10x**. This enables NVILA-HD-Video to efficiently scale to 4K-resolution, 1K-frame videos while achieving improved performance on benchmarks such as VideoMME and state-of-the-art performance on HLVid.

<hr style="border: 2px solid gray;"></hr>

## Pre-Trained Model

| Name | Parameters | Description | HuggingFace Link |
|------|------------|-------------|------------------|
| **NVILA-8B-HD-Video** | 8B | Video MLLM scaled to 4K resolution, 1K frames with AutoGaze | [nvidia/NVILA-8B-HD-Video](https://huggingface.co/nvidia/NVILA-8B-HD-Video) |

<hr style="border: 2px solid gray;"></hr>


## Quick Start

First install [AutoGaze](https://github.com/NVlabs/AutoGaze) following the instructions in its repo.

Then the following script provides a minimal example for how to use NVILA-HD-Video.

```python
import torch
from transformers import AutoModel, AutoProcessor

model_path = "nvidia/NVILA-8B-HD-Video"
video_path = "https://huggingface.co/datasets/bfshi/HLVid/resolve/main/example/clip_av_video_5_001.mp4"
prompt = "Question: What does the white text on the green road sign say?\n \
A. Hampden St\n \
B. Hampden Ave\n \
C. HampdenBlvd\n \
D. Hampden Rd\n \
Please answer directly with the letter of the correct answer."

# ----- Video processing args -----
num_video_frames = 128           # Total sampled frames for tiles
num_video_frames_thumbnail = 64  # Total sampled frames for thumbnails
max_tiles_video = 48             # Max spatial tiles per video (one tile is 392x392)

# ----- AutoGaze args (tiles) -----
gazing_ratio_tile = [0.2] + [0.06] * 15  # Per-frame max gazing ratios (single float or list). Videos with higher resolution/FPS usually need lower gazing ratio.
task_loss_requirement_tile = 0.6         # AutoGaze stops gazing at each frame when the estimated reconstruction loss of that frame is lower than this threshold.

# ----- AutoGaze args (thumbnails) -----
gazing_ratio_thumbnail = 1       # Set gazing ratio to 1 and task loss requirement to None to skip gazing on thumbnails
task_loss_requirement_thumbnail = None

# ----- Batching -----
max_batch_size_autogaze = 16     # Set AutoGaze and SigLIP to use smaller mini-batch size if GPU memory is limited
max_batch_size_siglip = 32

# Load processor and model
processor = AutoProcessor.from_pretrained(
    model_path,
    num_video_frames=num_video_frames,
    num_video_frames_thumbnail=num_video_frames_thumbnail,
    max_tiles_video=max_tiles_video,
    gazing_ratio_tile=gazing_ratio_tile,
    gazing_ratio_thumbnail=gazing_ratio_thumbnail,
    task_loss_requirement_tile=task_loss_requirement_tile,
    task_loss_requirement_thumbnail=task_loss_requirement_thumbnail,
    max_batch_size_autogaze=max_batch_size_autogaze,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    max_batch_size_siglip=max_batch_size_siglip,
)
model.eval()

# Run inference
video_token = processor.tokenizer.video_token
inputs = processor(text=f"{video_token}\n\n{prompt}", videos=video_path, return_tensors="pt")
inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

outputs = model.generate(**inputs)
response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
print(response)
```

<hr style="border: 2px solid gray;"></hr>

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{autogaze,
  title={AutoGaze},
  author={},
  year={2025}
}
```
