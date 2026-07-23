<div align="center">

## Causal Forcing & Causal Forcing++
### Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation

<p align="center">
  <p align="center">
    <div>
    <a href="https://zhuhz22.github.io/" target="_blank">Hongzhou Zhu*</a><sup></sup>,
    <a href="https://gracezhao1997.github.io/" target="_blank">Min Zhao*</a><sup></sup> , 
    <a href="https://guandehe.github.io/" target="_blank">Guande He</a><sup></sup>, 
    <a href="https://scholar.google.com/citations?user=dxN1_X0AAAAJ&hl=en" target="_blank">Hang Su</a><sup></sup>,
    <a href="https://zhenxuan00.github.io/" target="_blank">Chongxuan Li</a><sup></sup> ,
    <a href="https://ml.cs.tsinghua.edu.cn/~jun/index.shtml" target="_blank">Jun Zhu</a><sup></sup>
  </div>
  <div>
      <sup></sup>Tsinghua University & Shengshu & UT Austin & RUC
  </div>


</div>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2602.02214">Causal Forcing</a> | <a href="https://arxiv.org/abs/2605.15141">Causal Forcing++</a> | <a href="https://thu-ml.github.io/CausalForcing.github.io">Website</a> | <a href="https://huggingface.co/zhuhz22/Causal-Forcing/tree/main">Models</a> | <a href="https://my.feishu.cn/wiki/AjBSwcjpqiN0ECkodIWcGDcMn4e?from=from_copylink">Document</a> </h3>
</p>



-----
The Causal Forcing series uses **Causal ODE** or **Causal Consistency Distillation** to drive asymmetric DMD as a theoretically correct initialization for real-time interactive video generation. **Refer to [this extension for long video generation](#minute-level-long-video-generation).**

[Causal Forcing](https://arxiv.org/abs/2602.02214) significantly outperforms Self Forcing in **both visual quality and motion dynamics**, while keeping **the same training budget and inference efficiency**. We support both chunk-wise and **frame-wise** models, with the latter natively unifying T2V and **I2V**.

We further propose [**Causal Forcing++**](https://arxiv.org/abs/2605.15141)[ technical report ], replacing ODE with **causal Consistency Distillation** to eliminate ODE data curation and improve performance, releasing the first **1-step/2-step frame-wise** models.

-----
<img width="2090" height="850" alt="overview" src="assets/pipeline.png" />

## Table of Contents

- [Quick Start](#quick-start)
  - [Installation](#installation) 
  - [Inference: 1/2/4-step T2V & I2V](#cli-inference)
  - [Long Video](#minute-level-long-video-generation)
- [Training Pipeline](#training) (Causal Foricng / Causal Forcing++)
  - [Stage 1: AR Diffusion](#training)
  - [Stage 2: Causal ODE (Causal Forcing)](#training) / [🔥Causal CD (Causal Forcing++)](#-stage-2-option-b-causal-consistency-distillation-initialization-causal-forcing)
  - [Stage 3: Asymmetric DMD](#stage-3-dmd)


| Models              | Checkpoints | Description|
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------|
| Chunk-wise 4-step   | 🤗 [Huggingface](https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/chunkwise/causal_forcing.pt)    | SOTA AR model on Wan1.3B outperforming Self Forcing|
| Frame-wise 4-step   | 🤗 [Huggingface](https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/framewise/causal_forcing.pt)    | Rich dynamics and high quality |
| Frame-wise 2-step🔥 | 🤗 [Huggingface](https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/causal-forcing%2B%2B/framewise-2step.pt)    | The  first frame-wise 2-step model **even better than 4-step**|
| Frame-wise 1-step   | 🤗 [Huggingface](https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/causal-forcing%2B%2B/framewise-1step.pt)    | Extremely low latency |
| Long Video Generator  🔥 | 🤗 [Huggingface](https://huggingface.co/zhuhz22/Causal-Forcing/blob/main/chunkwise/longvideo.pt)    | Minute-long AR video generator |
| HY1.5-TI2V (8B) 🔥  | 🤗 [Huggingface](https://huggingface.co/MIN-Lab/minWM)   | Refer to [this repo](https://github.com/shengshu-ai/minWM). |
| Action-conditioned WM   | 🤗 [Huggingface](https://huggingface.co/MIN-Lab/minWM)    | Refer to [this repo](https://github.com/shengshu-ai/minWM). |

-----

https://github.com/user-attachments/assets/310f0cfa-e1bb-496d-8941-87f77b3271c0

## 🔥 News
- **2026.7.23**: [Self Gradient Forcing](https://github.com/zhuang2002/Self_Gradient_Forcing) is built on Causal Forcing initialization.
- **2026.7.20**: Happy to see that the recent SOTA video world models [DreamX-World 1.0](https://arxiv.org/pdf/2606.16993) and [Matrix-Game 3.5](https://matrix-game-v3-5.github.io/paper/Matrix-Game-3.5.pdf) are built on Causal Forcing!
- **2026.5.17**: We release Causal Forcing for the HY1.5-TI2V-8B model! Refer to [this repo](https://github.com/shengshu-ai/minWM) for the details. This model explicitly supports I2V.
- **2026.5.15**: We release [Causal Forcing++](https://arxiv.org/abs/2605.15141), supporting Casual Consistency Distillation for few-step initialization, and open-source **the first frame-wise 2-step AR model** comparable to chunk-wise 4-step models!
- **2026.5.10**: Thanks to @[AshadowZ](https://github.com/AshadowZ), now our chunk-wise ODE data curation is **3x faster**!
- **2026.4.16**: **Optimize the Stage 2 🔥consistency distillation🔥 infrastructure for 3× faster training, let's try it now!** We have also released the ckpt.
- **2026.3.15** : [Rolling Sink](https://github.com/haodong2000/RollingSink), [Infinity-RoPE](https://github.com/yesiltepe-hidir/infinity-rope) and [Deep Forcing](https://cvlab-kaist.github.io/DeepForcing/) adopt Causal Forcing as one of the base models!
- **2026.2.28** : Add [FAQ section](#faq--blog) regarding hot topics, specifically which is the better Initialization between AR diffusion and causal ODE distillation.
- **2026.2.11** : We now support **I2V** generation! Feel free to try it [here](#new-i2v)!
- **2026.2.7** : Causal Forcing now supports [Rolling Forcing](https://github.com/TencentARC/RollingForcing), enabling minute-level long video generation!
- **2026.2.5** : Release causal consistency distillation (Preview) as substitute for ODE distillation, **free of generating ODE paired data**!
- **2026.2.2** : The [paper](https://arxiv.org/abs/2602.02214), [project page](https://thu-ml.github.io/CausalForcing.github.io/), and code are released.


## Quick Start

> The inference environment is identical to Self Forcing.

**NOTE**: Similar to CausVid/Self Forcing, Causal Forcing does not natively support videos longer than 81 frames. As a base training method, it is orthogonal to techniques like Longlive/Rolling Forcing. To use Causal Forcing as a long video baseline, see [this extension](#minute-level-long-video-generation). **Directly using the 5-second trained Causal Forcing model as a baseline for long video generation is extremely unfair**.


### Installation
```bash
conda create -n causal_forcing python=3.10 -y
conda activate causal_forcing
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install flash-attn --no-build-isolation
python setup.py develop
```
### Download Checkpoints
```bash
hf download Wan-AI/Wan2.1-T2V-1.3B  --local-dir wan_models/Wan2.1-T2V-1.3B
hf download Wan-AI/Wan2.1-T2V-14B  --local-dir wan_models/Wan2.1-T2V-14B
# Causal Forcing
hf download zhuhz22/Causal-Forcing chunkwise/causal_forcing.pt --local-dir checkpoints
hf download zhuhz22/Causal-Forcing framewise/causal_forcing.pt --local-dir checkpoints
# Causal Forcing++
hf download zhuhz22/Causal-Forcing causal-forcing++/framewise-2step.pt --local-dir checkpoints
hf download zhuhz22/Causal-Forcing causal-forcing++/framewise-1step.pt --local-dir checkpoints
```

### CLI Inference

#### T2V
Chunk-wise model:
```bash
python inference.py \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --output_folder output/chunkwise \
  --checkpoint_path checkpoints/chunkwise/causal_forcing.pt \
  --data_path prompts/demos.txt
```

Frame-wise model:

```bash
# =============== Causal Forcing++ ================
# 2-step Causal Forcing++
python inference.py \
  --config_path configs/causal_forcing_dmd_framewise_2step.yaml \
  --output_folder output/framewise_2step_cf++ \
  --checkpoint_path checkpoints/causal-forcing++/framewise-2step.pt \
  --data_path prompts/demos.txt \
  --use_ema

# 1-step Causal Forcing++
python inference.py \
  --config_path configs/causal_forcing_dmd_framewise_1step.yaml \
  --output_folder output/framewise_1step_cf++ \
  --checkpoint_path checkpoints/causal-forcing++/framewise-1step.pt \
  --data_path prompts/demos.txt \
  --use_ema

# =============== Causal Forcing ================
# 4-step Causal Forcing
python inference.py \
  --config_path configs/causal_forcing_dmd_framewise.yaml \
  --output_folder output/framewise \
  --checkpoint_path  checkpoints/framewise/causal_forcing.pt \
  --data_path prompts/demos.txt \
  --use_ema 
```

#### I2V
> Our frame-wise setting natively supports I2V. You simply need to set the first latent initial frame as your conditional image. 

```bash
python inference.py \
  --config_path configs/causal_forcing_dmd_framewise.yaml \
  --output_folder output/framewise \
  --checkpoint_path  checkpoints/framewise/causal_forcing.pt \
  --data_path prompts/i2v \
  --i2v \
  --use_ema
```


### Minute-level Long Video Generation
Built on [Rolling Forcing](https://github.com/TencentARC/RollingForcing), we implemented minute-level long video generation. See [here](./long_video) for the detail.

[Infinity-RoPE](https://github.com/yesiltepe-hidir/infinity-rope), [Deep Forcing](https://cvlab-kaist.github.io/DeepForcing/) and [Rolling Sink](https://github.com/haodong2000/RollingSink) also adopt Causal Forcing as one of their base models, enabling interactive (prompt-switchable) long video generation at the minute scale. You can also try them out at their repos.

## Training

<details>
<summary> Stage 1: Autoregressive Diffusion Training (Can skip by using our pretrained checkpoints. Click to expand.)</summary>

First download the dataset (we provide a 6K toy dataset here):
```bash
hf download zhuhz22/Causal-Forcing-data  --local-dir dataset
python utils/merge_and_get_clean.py
```
> If the download gets stuck, Ctrl^C and then resume it.


> For training on your own dataset, refer to [this issue](https://github.com/thu-ml/Causal-Forcing/issues/8).


Then train the AR-diffusion model:
- Framewise:
  ```bash
    torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/ar_diffusion_tf_framewise.yaml \
    --logdir logs/ar_diffusion_framewise
  ```

- Chunkwise:
  ```bash
    torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/ar_diffusion_tf_chunkwise.yaml \
    --logdir logs/ar_diffusion_chunkwise
  ```

> We recommend training no less than 2K steps, and more steps (e.g., 5~10K) will lead to better performance.

Inference to test training results:
```bash
python inference.py \
  --config_path configs/ar_diffusion_tf_{framewise OR chunkwise}.yaml \
  --output_folder output/{framewise OR chunkwise}_ar_diffusion \
  --checkpoint_path  checkpoints/{framewise OR chunkwise}/ar_diffusion.pt \
  --data_path prompts/demos.txt
```
</details>


<details>
<summary> Stage 2 (Option a): Causal ODE Initialization (Can skip by using our pretrained checkpoints. Click to expand.)</summary>

🔥 Thanks to @[AshadowZ](https://github.com/AshadowZ), now our chunk-wise ODE data curation is **3x faster**!

You can also use `bf16` to accelerate generation.

If you have skipped Stage 1, you need to download the pretrained models:
```bash
hf download zhuhz22/Causal-Forcing framewise/ar_diffusion.pt --local-dir checkpoints
hf download zhuhz22/Causal-Forcing chunkwise/ar_diffusion.pt --local-dir checkpoints
```

In this stage, first generate ODE paired data:
```bash
# for the frame-wise model
torchrun --nproc_per_node=8 \
  get_causal_ode_data_framewise.py \
  --generator_ckpt checkpoints/framewise/ar_diffusion.pt \
  --rawdata_path dataset/clean_data \
  --output_folder dataset/ODE6KCausal_framewise_latents

python utils/create_lmdb_iterative.py \
  --data_path dataset/ODE6KCausal_framewise_latents \
  --lmdb_path dataset/ODE6KCausal_framewise

# for the chunk-wise model
torchrun --nproc_per_node=8 \
  get_causal_ode_data_chunkwise.py \
  --generator_ckpt checkpoints/chunkwise/ar_diffusion.pt \
  --rawdata_path dataset/clean_data \
  --output_folder dataset/ODE6KCausal_chunkwise_latents

python utils/create_lmdb_iterative.py \
  --data_path dataset/ODE6KCausal_chunkwise_latents \
  --lmdb_path dataset/ODE6KCausal_chunkwise

#🔥NEW: Or you can use the optimized code (3x speedup) by @AshadowZ 
torchrun --nproc_per_node=8 \
  get_causal_ode_data_kv.py \
  --generator_ckpt checkpoints/{chunkwise,framewise}/ar_diffusion.pt \
  --rawdata_path /mnt/vepfs/base2/zhaomin/group_0001_lmdb \
  --output_folder dataset/ODE6KCausal_{chunkwise,framewise}_latents \
  --num_frames_per_chunk 3/1 # 3 for chunkwise, 1 for framewise
  --generation_mode blockwise_kv
```

Or you can also directly download our prepared dataset (~300G):
```bash
hf download zhuhz22/Causal-Forcing-data  --local-dir dataset
python utils/merge_lmdb.py
```
> If the download gets stuck, Ctrl^C and then resume it.


And then train ODE initialization models:
- Frame-wise:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_ode_framewise.yaml \
    --logdir logs/causal_ode_framewise
  ```
- Chunk-wise:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_ode_chunkwise.yaml \
    --logdir logs/causal_ode_chunkwise
  ```

> We recommend training no less than 1K steps, and more steps (e.g., 5~10K) will lead to better performance.

Inference to test training results:

The same as [here](#cli-inference).
</details>



### 🔥 Stage 2 (Option b): Causal Consistency Distillation Initialization ([Causal Forcing++](https://arxiv.org/abs/2605.15141))
Since creating ODE-paired data is very time-consuming, we also provide an alternative here that achieves the same effect (or better) as ODE distillation while requiring only ground-truth data, **free of generating ODE data!**

> Thanks to [@chijw's effort](https://github.com/thu-ml/Causal-Forcing/pull/20), now the EMA mechanism is more efficient!

If you have skipped Stage 1, you need to download the pretrained multi-step AR diffusion models:
```bash
hf download zhuhz22/Causal-Forcing framewise/ar_diffusion.pt --local-dir checkpoints
hf download zhuhz22/Causal-Forcing chunkwise/ar_diffusion.pt --local-dir checkpoints
```

- Frame-wise:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_cd_framewise.yaml \
    --logdir logs/causal_cd_framewise
  ```
- Chunk-wise:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_cd_chunkwise.yaml \
    --logdir logs/causal_cd_chunkwise
  ```

> We recommend training no less than 3K steps, and more steps (e.g., 5~10K) will lead to better performance.

<details>
<summary>
You can also download the checkpoints directly (click to expand) :
</summary>

```bash
hf download zhuhz22/Causal-Forcing framewise/causal_cd.pt --local-dir checkpoints
hf download zhuhz22/Causal-Forcing chunkwise/causal_cd.pt --local-dir checkpoints
```
</details>

Inference to test training results: the same as [here](#cli-inference).




### Stage 3: DMD

First download the dataset:
```bash
hf download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
```

<details>

<summary>
If you have skipped Stage 2, you need to download our pretrained checkpoints (click to expand):
</summary>

```bash
hf download zhuhz22/Causal-Forcing framewise/causal_ode.pt --local-dir checkpoints
hf download zhuhz22/Causal-Forcing chunkwise/causal_ode.pt --local-dir checkpoints

# 🔥 Or you can also use the improved version (Causal Forcing++): causal CD instead of causal ODE
hf download zhuhz22/Causal-Forcing framewise/causal_cd.pt --local-dir checkpoints
hf download zhuhz22/Causal-Forcing chunkwise/causal_cd.pt --local-dir checkpoints
```
</details>

And then train DMD models:

- Frame-wise model:
  ```bash
  # =============== Causal Forcing ================
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_forcing_dmd_framewise.yaml \
    --logdir logs/causal_forcing_dmd_framewise
  # =============== Causal Forcing++ ================
  # 2-step Causal Forcing++
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_forcing_dmd_framewise_2step.yaml \
    --logdir logs/causal_forcing_dmd_framewise

  # 1-step Causal Forcing++
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_forcing_dmd_framewise_1step.yaml \
    --logdir logs/causal_forcing_dmd_framewise
  ```

- Chunk-wise model:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_forcing_dmd_chunkwise.yaml \
    --logdir logs/causal_forcing_dmd_chunkwise
  ```
> For bs64, we recommend training for no more than 1K steps; otherwise the motion dynamics may degrade. If bs is smaller (e.g., 8), more training steps is preferable.

Such models are the final models used to generate videos.
## FAQ & Blog 
See the [FAQ](https://my.feishu.cn/wiki/AjBSwcjpqiN0ECkodIWcGDcMn4e) and the [blog](https://zhuanlan.zhihu.com/p/2002114039493461457). (currently in Chinese)

<details>
<summary> Typical Questions (click to expand)
</summary>

**Why using bidirectional teacher in the DMD stage ?**
- Q: In the DMD stage, do you still use a bidirectional teacher? Why not an AR teacher?
- A: Yes. DMD only requires the student to match the teacher’s final distribution, not the generation trajectory, so a bidirectional teacher is fine. Also, bidirectional diffusion models are typically stronger than AR diffusion, so they make a better teacher.
  
- Q: Then why must the ODE (or Consistency Distillation) stage use an AR teacher?
- A: Because ODE/CD requires the student and teacher to follow the same trajectory, so their structures must be matched; an AR student cannot be trajectory-aligned with a bidirectional teacher.
  
🔥🔥 **ODE initialization or multi-step AR diffusion initialization ?**
- Q: Which is better as initialization: a “proper” ODE initialization or directly using multi-step AR diffusion?
- A: We compared this in the Appendix C2. Overall, proper ODE init is better: multi-step AR diffusion init + DMD occasionally yields grid-like or waxy/greasy results. A key reason is that DMD is inherently few-step, so the right comparison is under few-step; in that regime, a few-step diffusion teacher is much weaker than an ODE-distilled teacher. Without ODE distillation, DMD must both close the step gap and handle an added conditioning gap from self-rollout: early few-step errors corrupt the history and get amplified across frames (large exposure bias), which increases DMD pressure. It can still converge, but typically with worse quality than ODE initialization. Also, with ODE init, DMD can be trained very few steps (e.g., ~100), reducing the risk of dynamics degradation from long DMD training.
  
**Can frame-level non-injectivity appears in the actual training dataset ?**
- Q: Regarding the “one-to-many” analysis in the ODE stage: since a single frame’s latent has very high dimensionality, isn’t the probability of being exactly identical extremely small?
- A: Yes, but the key point here is not whether the dataset literally contains identical samples; it’s whether there exists a well-defined function in the mathematical sense. Our vision modalities live in a continuous space—even in 1D, getting two samples to be exactly identical is extremely unlikely. However, the theoretical existence of exact collisions is enough to break the function property and make it ill-defined.

</details>

## Acknowledgements
- This codebase is built on top of the open-source implementation of [CausVid](https://github.com/tianweiy/CausVid), [Self Forcing](https://github.com/guandeh17/Self-Forcing), [Rolling Forcing](https://github.com/TencentARC/RollingForcing) and the [Wan2.1](https://github.com/Wan-Video/Wan2.1) repo. 
- Thanks to @[chijw](https://github.com/chijw) for improving the EMA mechanism. 
- Thanks to @[AshadowZ](https://github.com/AshadowZ) for improving the causal ODE data curation efficiency.
- Causal Forcing++ with 1/2 steps applies the first-frame 4-step technique proposed by [ASD](https://github.com/BigAandSmallq/SAD). We thank ASD for its contribution.

## References
If you find the method useful, please cite
```
@article{zhu2026causal,
  title={Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation},
  author={Zhu, Hongzhou and Zhao, Min and He, Guande and Su, Hang and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2602.02214},
  year={2026}
}

@article{zhao2026causal,
  title={Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation},
  author={Zhao, Min and Zhu, Hongzhou and Zheng, Kaiwen and Zhou, Zihan and Yan, Bokai and Li, Xinyuan and Yang, Xiao and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2605.15141},
  year={2026}
}
```
