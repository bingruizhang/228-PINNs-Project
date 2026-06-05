# ECE 228 Final Project Detailed Handoff

本文档是给队友接手项目用的完整交接说明。它的目标不是简单列几个命令，而是让任何一个组员从这个文件开始，就能理解：

- 我们为什么把项目从原 proposal 改成现在这个方向
- 整个文件夹每一部分分别是什么
- 代码如何组织
- 实验如何运行
- 结果在哪里看
- 报告和 presentation 应该怎么写
- 哪些结论可以放心使用，哪些地方写作时要谨慎

项目根目录是：

```text
C:\Users\zbrrrrr\Desktop\228-clean
```

## 1. 当前项目状态

原 proposal 的方向是使用 `pinns-torch` 跑一个 Navier-Stokes PINN 示例。老师反馈说这个方向本身可以，但如果只是训练一个 fully connected network，再做一些 `n_train`、层数、epoch 之类的 hyperparameter tuning，会显得太基础，接近普通回归任务。

因此我们把项目升级成：

```text
Comparing PINNs, Coordinate Networks, and Neural Operators for Inverse Navier-Stokes Learning
```

核心问题变成：

```text
对于 Navier-Stokes 流场学习和物理参数反演，不同网络结构到底谁更适合？
```

现在项目已经完成了以下内容：

- 自包含 PyTorch 实验框架已经搭好，位置在 `src/pinn_project/`
- 多个模型已经实现：FCN、Deep FCN、Fourier PINN、SIREN PINN、Residual PINN、FNO
- 完整实验 suite 已经跑完
- SIREN 根据 ablation 结果做了优化重跑
- 最终结果表已经生成：`results/final_summary.csv`
- 汇总图已经生成：`results/figures/`
- report 草稿、presentation 草稿、结果总结都已经写好

当前正式实验共有 68 条记录。最终主要结果见：

```text
docs/final_results_summary.md
results/final_summary.csv
results/figures/architecture_summary.png
```

## 2. 环境说明

本项目运行时使用本地 conda 环境：

```powershell
5080py
```

本地 GPU 是 RTX 5080，PyTorch 已确认可以识别 CUDA。

在项目根目录运行：

```powershell
conda run -n 5080py python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期输出类似：

```text
torch 2.9.1+cu130
cuda True
NVIDIA GeForce RTX 5080
```

运行项目模块前，一定要在 PowerShell 里设置：

```powershell
$env:PYTHONPATH = "src"
```

否则 `python -m pinn_project.train` 可能找不到模块。

推荐所有命令都从项目根目录运行：

```powershell
cd C:\Users\zbrrrrr\Desktop\228-clean
```

## 3. 顶层目录结构总览

项目根目录目前主要包括：

```text
228-clean/
  README.md
  requirements.txt
  generate_pdf.py
  ECE_228_Lecture_1__2026_Spring_.pdf
  docs/
  experiments/
  results/
  src/
  notebooks/
  data/
```

每个部分的作用如下。

### 3.1 `README.md`

这是项目的简版说明文件，适合给 GitHub 或 TA 看。里面包括：

- 项目 idea
- 包含的模型
- 包含的实验
- 环境命令
- smoke test 命令
- full suite 命令
- 结果输出位置
- report 主线

如果要给外部人员快速介绍项目，先让对方看 `README.md`。

### 3.2 `requirements.txt`

这是原始 `pinns-torch` 风格的依赖文件，里面包括：

- `torch`
- `torchvision`
- `lightning`
- `torchmetrics`
- `hydra-core`
- `scipy`
- `matplotlib`
- `pyDOE`
- 其他辅助工具

注意：我们最终新写的 `src/pinn_project` 框架主要依赖 PyTorch、NumPy、SciPy、Matplotlib、CSV/JSON 标准库等。因为 `5080py` 环境已经能跑完整项目，所以现在不需要重新安装依赖。

### 3.3 `generate_pdf.py`

这是 proposal 阶段用来生成 PDF 的小脚本。它不是 final project 实验的核心代码。

### 3.4 `ECE_228_Lecture_1__2026_Spring_.pdf`

这是课程第一讲 PDF，里面包含课程安排、project 要求和课程 topic。它主要用于证明项目和课程内容相关：

- Week 7: Neural Operator
- Week 8: PINN
- Week 10: Final project presentation

写 report 或 presentation 时可以提到我们项目对应 Week 7 和 Week 8。

### 3.5 `data/`

这个目录用于说明和保存数据相关内容。我们最终自包含实验使用 analytic Taylor-Green vortex，不依赖外部下载数据。

也就是说，最终结果不是从 `data/` 读大数据集，而是在 `src/pinn_project/data.py` 中直接生成解析解。为了避免队友误会“是不是漏放数据集”，现在 `data/` 里也放了一个小的预生成样本：

```text
data/README.md
data/taylor_green_sample.npz
```

`taylor_green_sample.npz` 只用于快速查看 benchmark 数组，不是训练必需文件。

### 3.6 `notebooks/`

当前包含：

```text
notebooks/tutorials/0-Schrodinger.ipynb
```

这是原始 `pinns-torch` 相关 notebook。它不是最终主实验，但可作为 fallback 或参考材料。

### 3.7 `src/`

这是源代码主目录，分成两块：

```text
src/examples/
src/pinn_project/
```

`src/examples/` 是原始 `pinns-torch` 示例代码，保留为 proposal 来源和参考。

`src/pinn_project/` 是我们这次 final project 新写的核心实验框架。

### 3.8 `experiments/`

这里放所有 PowerShell 实验脚本。队友要复现实验，基本从这里开始。

### 3.9 `results/`

这里放所有已经跑出来的实验结果，包括：

- 每个 run 的 `metrics.json`
- 每个 run 的 `history.csv`
- 每个 run 的 `loss_curve.png`
- coordinate PINN 的 `slice_t050.png`
- 汇总表 `final_summary.csv`
- 汇总图 `results/figures/`

## 4. `docs/` 文件夹详细说明

`docs/` 是文档和写作材料目录。当前包括：

```text
docs/
  proposal.pdf
  proposal.tex
  proposal_draft.md
  review_summary.md
  project_handoff.md
  final_results_summary.md
  final_report_plan.md
  final_report_draft.md
  presentation_plan.md
  final_presentation_script.md
```

### 4.1 `proposal_draft.md`

这是最初 proposal 的 Markdown 草稿。内容包括：

- Problem Background & Motivation
- Related Work
- High-level Methodology
- References

它保留了我们最初的项目构想：用 PINN 解 Navier-Stokes。现在 final project 已经大幅扩展，所以写最终报告时不要直接照抄这个 proposal。

### 4.2 `proposal.tex`

这是 proposal 的 LaTeX 版本。它用于生成 proposal PDF。

### 4.3 `proposal.pdf`

这是提交 proposal 时用的 PDF 文件。

### 4.4 `review_summary.md`

这是 proposal 阶段整理给组内审阅用的 review summary，记录当时的任务要求和 proposal 设计逻辑。

### 4.5 `project_handoff.md`

就是当前这个文件。它是最完整的交接文件。

队友如果只想看一个文档来接手项目，就先看这个。

### 4.6 `final_results_summary.md`

这是最终实验结果总结，重点包括：

- 哪些实验已经跑完
- 架构比较表格
- main takeaways
- 哪些图适合放 report
- 写作时要注意什么

它是写 Results & Analysis 最重要的参考文件之一。

### 4.7 `final_report_plan.md`

这是 final report 的结构规划。它不是完整正文，而是告诉我们 6-8 页报告怎么组织：

- Introduction
- Related Work
- Method
- Experiments
- Results and Analysis
- Conclusion

如果队友负责写报告大纲，可以先看这个。

### 4.8 `final_report_draft.md`

这是已经写好的 final report 草稿正文。它包括：

- Abstract
- Introduction
- Related Work
- Problem Setup
- Method
- Experiments
- Results
- Discussion
- Limitations and Future Work
- Conclusion

它可以作为最终报告的初稿。后续要做的是：

- 转成 NeurIPS/课程要求格式
- 插入图表
- 压缩或扩写到 6-8 页
- 检查引用
- 调整语言让它更像学生完成的项目报告

### 4.9 `presentation_plan.md`

这是 3 分钟 highlight presentation 的 slide 规划。它告诉我们每页 slide 应该放什么。

### 4.10 `final_presentation_script.md`

这是 3 分钟 presentation 的讲稿草稿。队友可以直接根据它做 slides 和练习。

## 5. `src/examples/` 原始参考代码说明

`src/examples/` 保留了原始 `pinns-torch` 示例。它们不是我们最终新框架的主代码，但对项目背景有用。

当前包括：

```text
src/examples/ac/
src/examples/aneurysm3D/
src/examples/burgers_continuous_forward/
src/examples/burgers_continuous_inverse/
src/examples/burgers_discrete_forward/
src/examples/burgers_discrete_inverse/
src/examples/kdv/
src/examples/navier_stokes/
src/examples/schrodinger/
```

每个 example 通常包括：

```text
README.md
train.py
configs/config.yaml
```

### 5.1 `src/examples/navier_stokes/`

这是 proposal 里最重要的原始参考。里面说明了 continuous inverse Navier-Stokes 问题：

- 输入：`t, x, y`
- 输出：`psi, p`
- 通过 `psi` 计算 `u, v`
- 同时学习 unknown parameters `lambda_1, lambda_2`
- 使用 PDE residual

这个 example 支撑了我们 final project 的 inverse Navier-Stokes 叙事。

但最终实验没有直接依赖 `pinnstorch` 包，因为 `5080py` 环境里一开始没有安装 `pinnstorch`。为了保证可复现，我们新写了自包含 PyTorch 框架。

### 5.2 `src/examples/burgers_continuous_inverse/`

这是 inverse Burgers' equation 参考。它说明 PINN 可以同时预测解和学习 PDE 参数。

可以在 report 相关工作或 future work 里提到：我们的 inverse Navier-Stokes 设置和 inverse Burgers 设置思想相似，但流体问题更复杂。

### 5.3 `src/examples/schrodinger/`

这是 Schrodinger equation 示例。它原来是 proposal fallback plan 的参考。

现在不再作为主实验，因为我们已经完成了更完整的 Navier-Stokes benchmark。

### 5.4 其他 examples

其他目录如 `ac`、`kdv`、`aneurysm3D`、`burgers_discrete_*` 都是原始 PINNs-Torch 示例。它们可以作为背景材料，但 final report 不需要逐个介绍。

## 6. `src/pinn_project/` 核心代码详细说明

这是最终项目最重要的代码目录。

```text
src/pinn_project/
  __init__.py
  data.py
  models.py
  physics.py
  metrics.py
  plotting.py
  train.py
  summarize.py
  make_figures.py
```

### 6.1 `__init__.py`

标记 `pinn_project` 是一个 Python package。内容很少，但必要。

### 6.2 `data.py`

负责数据和解析解。

主要内容：

- `Domain`
- `taylor_green_solution`
- `sample_coords`
- `grid_coords`
- `add_noise`

#### `Domain`

定义空间和时间范围：

```text
x in [0, 2pi]
y in [0, 2pi]
t in [0, 1]
```

#### `taylor_green_solution(coords, nu=0.01)`

生成 Taylor-Green vortex 的解析解：

```text
u = -cos(x) sin(y) exp(-2 nu t)
v =  sin(x) cos(y) exp(-2 nu t)
p = -0.25 [cos(2x) + cos(2y)] exp(-4 nu t)
```

输出是：

```text
[u, v, p]
```

为什么用 Taylor-Green：

- 有 exact ground truth
- 有 exact PDE parameters
- 可以做很多 controlled experiments
- 不依赖外部大数据
- 适合比较 architecture

#### `sample_coords`

随机采样 `(x, y, t)` 点，用于：

- observation data
- collocation points

#### `grid_coords`

生成规则网格，用于：

- evaluation
- heatmap visualization
- FNO 输入/输出

#### `add_noise`

给数据加噪声，用于 noise robustness experiment。

### 6.3 `models.py`

负责所有模型结构。

当前模型包括：

- `MLP`
- `FourierMLP`
- `ResidualMLP`
- `SIREN`
- `TinyFNO2d`
- `make_model`

#### `normalize_coords`

把坐标归一化到大致 `[-1, 1]`：

```text
x: [0, 2pi] -> [-1, 1]
y: [0, 2pi] -> [-1, 1]
t: [0, 1] -> [-1, 1]
```

这个很重要，因为不归一化会让网络训练更不稳定。

#### `MLP`

普通 fully connected network，也就是 vanilla FCN PINN baseline。

模型输入：

```text
(x, y, t)
```

模型输出：

```text
(u, v, p)
```

#### `FourierMLP`

先对坐标做 Fourier features，再输入 MLP。

对应老师反馈里的：

```text
input-conditioned prediction / specialized network
```

它适合讨论 coordinate representation，但最终结果显示它的 field error 可以接受，物理参数恢复不够好。

#### `ResidualMLP`

带 residual blocks 的 MLP。

这是最终结果最强的完整 inverse-PINN 模型。它同时做到：

- 低 velocity error
- 低 pressure error
- 低 PDE residual
- 好的 `lambda_1, lambda_2` 恢复

报告里应该重点讲 Residual PINN。

#### `SIREN`

sinusoidal representation network。适合连续信号和导数，但训练很敏感。

我们第一次 full suite 用较深 SIREN 时表现不稳定，后来通过 ablation 发现浅层 SIREN 更好，所以重新跑了优化后的 SIREN。

这个可以写成一个有价值的 insight：

```text
SIREN has strong field representation ability, but is sensitive to depth and optimization.
```

#### `TinyFNO2d`

小型 Fourier Neural Operator 风格模型。

注意：当前 FNO 是 field-prediction baseline，不是完整 inverse PINN。

它：

- 输入 grid/time 信息
- 输出整张场 `[u, v, p]`
- 不使用 PDE residual
- 不学习 `lambda_1, lambda_2`

所以 report 里不能说 FNO 也完成了 physical parameter recovery。正确写法是：

```text
FNO is a neural-operator-style field prediction baseline.
```

#### `make_model`

根据字符串创建模型：

```text
fcn
deep_fcn
fourier
siren
residual
fno
```

训练脚本通过 `--model` 参数调用它。

### 6.4 `physics.py`

负责物理 residual。

主要内容：

- `grad`
- `navier_stokes_residual`

#### `grad`

封装 PyTorch autograd，用于计算：

```text
u_x, u_y, u_t
v_x, v_y, v_t
p_x, p_y
u_xx, u_yy
v_xx, v_yy
```

#### `navier_stokes_residual`

计算 2D incompressible Navier-Stokes residual：

```text
f_u = u_t + lambda_1 (u u_x + v u_y) + p_x - lambda_2 (u_xx + u_yy)
f_v = v_t + lambda_1 (u v_x + v v_y) + p_y - lambda_2 (v_xx + v_yy)
continuity = u_x + v_y
```

训练时 loss 包含：

- `f_u`
- `f_v`
- `continuity`

### 6.5 `metrics.py`

负责误差指标。

主要内容：

- `relative_l2`
- `channel_metrics`

`channel_metrics` 会分别计算：

```text
rel_l2_u
rel_l2_v
rel_l2_p
```

这些指标是结果表里最主要的 field prediction metrics。

### 6.6 `plotting.py`

负责画图。

主要内容：

- `plot_slice`
- `plot_history`

#### `plot_slice`

生成 `slice_t050.png`，包含：

- truth u/v/p
- prediction u/v/p
- absolute error u/v/p

这类图适合放 report 的 qualitative results。

#### `plot_history`

生成 `loss_curve.png`，显示：

- total loss
- data loss
- PDE loss
- continuity loss

这类图适合说明训练过程。

### 6.7 `train.py`

这是单次实验最重要的入口。

运行格式：

```powershell
$env:PYTHONPATH = "src"
conda run -n 5080py python -m pinn_project.train --model residual --run-name demo_residual --steps 3000 --n-obs 5000
```

主要功能：

- 解析命令行参数
- 创建模型
- 生成 observation points
- 生成 collocation points
- 训练模型
- 计算 metrics
- 保存 `metrics.json`
- 保存 `history.csv`
- 保存 `loss_curve.png`
- 保存 `slice_t050.png`
- 保存 `checkpoint.pt`

重要参数：

```text
--model              选择模型：fcn/deep_fcn/fourier/siren/residual/fno
--run-name           本次实验的名字
--steps              训练步数
--n-obs              observation points 数量
--data-batch         data batch size
--colloc-batch       collocation batch size
--noise-std          噪声比例
--train-t-max        训练使用的最大时间，用于 time extrapolation
--width              网络宽度
--depth              网络深度
--lr                 learning rate
--pde-weight         PDE loss 权重
--continuity-weight  continuity loss 权重
--pde-warmup-steps   PDE loss warmup 步数
```

注意：

- PINN 类模型会学习 `lambda_1` 和 `lambda_2`
- `lambda_1, lambda_2` 使用 softplus 正值参数化，避免负 viscosity
- FNO 不学习 `lambda`
- FNO 不保存 `slice_t050.png`，因为它不是 coordinate model 的同一套 plotting path

### 6.8 `summarize.py`

负责把所有 run 的 `metrics.json` 合并成 CSV。

运行：

```powershell
conda run -n 5080py python -m pinn_project.summarize --results-dir results --out results/final_summary.csv --exclude-prefix smoke_
```

输出：

```text
results/final_summary.csv
```

`--exclude-prefix smoke_` 的作用是排除 smoke test 的短测试结果，避免污染正式 summary。

### 6.9 `make_figures.py`

负责从 `final_summary.csv` 生成汇总图。

运行：

```powershell
conda run -n 5080py python -m pinn_project.make_figures --summary results/final_summary.csv --out-dir results/figures
```

输出：

```text
results/figures/architecture_summary.png
results/figures/data_scarcity.png
results/figures/noise_robustness.png
results/figures/time_generalization.png
```

## 7. `experiments/` 实验脚本详细说明

当前包括：

```text
experiments/
  run_smoke.ps1
  run_preliminary_suite.ps1
  run_full_suite.ps1
  rerun_siren_optimized.ps1
```

### 7.1 `run_smoke.ps1`

用途：快速 sanity check。

它会短训练每个模型，确认：

- 代码能跑
- GPU 能用
- 每个模型 forward/backward 没问题
- metrics 能保存
- plots 能保存

运行：

```powershell
powershell -ExecutionPolicy Bypass -File experiments\run_smoke.ps1
```

输出目录：

```text
results/smoke_fcn/
results/smoke_fourier/
results/smoke_siren/
results/smoke_residual/
results/smoke_fno/
results/smoke_summary.csv
```

什么时候用：

- 修改代码后先跑它
- 不要一上来跑 full suite

### 7.2 `run_preliminary_suite.ps1`

用途：中等规模实验。

它比 smoke 更有意义，但比 full suite 快。适合：

- 快速看趋势
- 检查图表生成
- 调试实验矩阵

运行：

```powershell
powershell -ExecutionPolicy Bypass -File experiments\run_preliminary_suite.ps1
```

它会跑：

- architecture comparison
- data scarcity 的小版本
- noise robustness 的小版本
- time extrapolation 的小版本
- FNO resolution 的小版本

### 7.3 `run_full_suite.ps1`

用途：正式最终实验。

运行：

```powershell
powershell -ExecutionPolicy Bypass -File experiments\run_full_suite.ps1
```

它会跑：

1. Architecture comparison
2. Data scarcity study
3. Noise robustness study
4. Time extrapolation study
5. Neural operator resolution study
6. Fourier/SIREN representation ablations
7. Summary CSV generation
8. Summary figure generation

当前 full suite 已经跑过一次，结果已经在 `results/`。

如果之后没有改模型代码，不需要重新跑。

### 7.4 `rerun_siren_optimized.ps1`

用途：只重跑 SIREN 优化版。

背景：

第一次 full suite 里，深层 SIREN 训练不稳定。后来通过 ablation 发现 shallow SIREN 表现更好。因此我们新增这个脚本，只重跑 SIREN 相关实验，避免重复跑所有模型。

运行：

```powershell
powershell -ExecutionPolicy Bypass -File experiments\rerun_siren_optimized.ps1
```

它会覆盖：

- `arch_siren`
- `data_siren_*`
- `noise_siren_*`
- `time_siren_*`

并重新生成：

```text
results/final_summary.csv
results/figures/
```

## 8. `results/` 结果目录详细说明

`results/` 是所有训练输出。它包含两类内容：

1. 单个实验 run 的目录
2. 总结性结果文件

### 8.1 总结性结果文件

```text
results/final_summary.csv
results/smoke_summary.csv
results/figures/
```

#### `results/final_summary.csv`

这是最重要的最终结果表。

它包含 68 条正式实验记录，每条记录包括：

```text
model
run_name
n_obs
noise_std
train_t_max
steps
rel_l2_u
rel_l2_v
rel_l2_p
lambda_1
lambda_2
lambda_1_abs_error
lambda_2_abs_error
pde_residual_mse
continuity_mse
train_seconds
peak_gpu_mb
```

写 report 的定量表格主要从这里提取。

#### `results/smoke_summary.csv`

这是 smoke test 的结果表。它只用于确认代码能跑，不要放进 final report。

#### `results/figures/`

这里是 summary-level 图。

当前包括：

```text
architecture_summary.png
data_scarcity.png
noise_robustness.png
time_generalization.png
```

其中 `architecture_summary.png` 最适合 presentation。

### 8.2 单个 run 目录格式

每个实验 run 通常有一个目录，例如：

```text
results/arch_residual/
```

里面通常包括：

```text
metrics.json
history.csv
loss_curve.png
slice_t050.png
checkpoint.pt
```

注意：有些 FNO run 没有 `slice_t050.png` 或 checkpoint 文件可能不同，这是因为 FNO 使用 grid prediction path，不是 coordinate plotting path。

#### `metrics.json`

单次实验最终 metrics。

例如包括：

```text
rel_l2_u
rel_l2_v
rel_l2_p
lambda_1_abs_error
lambda_2_abs_error
pde_residual_mse
continuity_mse
train_seconds
peak_gpu_mb
```

#### `history.csv`

训练过程日志，按 `log_every` 间隔记录：

```text
step
loss
data_loss
pde_loss
continuity_loss
lambda_1
lambda_2
pde_warmup
```

#### `loss_curve.png`

训练 loss 曲线。可以用于 report appendix 或 debugging。

#### `slice_t050.png`

在 `t = 0.5` 的 field visualization，通常包含：

- truth
- prediction
- absolute error

对应 `u`、`v`、`p` 三个 channel。

#### `checkpoint.pt`

模型 checkpoint。用于后续恢复模型或继续分析。

### 8.3 已有结果目录分类

#### Architecture comparison

```text
results/arch_fcn/
results/arch_deep_fcn/
results/arch_fourier/
results/arch_siren/
results/arch_residual/
results/arch_fno/
```

这些是最重要的主实验。

#### Data scarcity

```text
results/data_fcn_n500/
results/data_fcn_n1000/
results/data_fcn_n2500/
results/data_fcn_n5000/
results/data_fcn_n10000/

results/data_fourier_n500/
results/data_fourier_n1000/
results/data_fourier_n2500/
results/data_fourier_n5000/
results/data_fourier_n10000/

results/data_siren_n500/
results/data_siren_n1000/
results/data_siren_n2500/
results/data_siren_n5000/
results/data_siren_n10000/

results/data_residual_n500/
results/data_residual_n1000/
results/data_residual_n2500/
results/data_residual_n5000/
results/data_residual_n10000/
```

用途：分析 observation points 数量变化对模型表现的影响。

#### Noise robustness

```text
results/noise_fcn_0/
results/noise_fcn_0p01/
results/noise_fcn_0p05/
results/noise_fcn_0p1/

results/noise_fourier_0/
results/noise_fourier_0p01/
results/noise_fourier_0p05/
results/noise_fourier_0p1/

results/noise_siren_0/
results/noise_siren_0p01/
results/noise_siren_0p05/
results/noise_siren_0p1/

results/noise_residual_0/
results/noise_residual_0p01/
results/noise_residual_0p05/
results/noise_residual_0p1/
```

用途：分析加噪声后模型是否仍然稳定。

#### Time extrapolation

```text
results/time_fcn_0p35/
results/time_fcn_0p5/
results/time_fcn_0p75/
results/time_fcn_1/

results/time_fourier_0p35/
results/time_fourier_0p5/
results/time_fourier_0p75/
results/time_fourier_1/

results/time_siren_0p35/
results/time_siren_0p5/
results/time_siren_0p75/
results/time_siren_1/

results/time_residual_0p35/
results/time_residual_0p5/
results/time_residual_0p75/
results/time_residual_1/
```

用途：训练只看到早期时间窗口，测试时看更完整时间范围。这个实验支持 generalization discussion。

#### FNO resolution

```text
results/resolution_fno_g24/
results/resolution_fno_g32/
results/resolution_fno_g48/
results/resolution_fno_g64/
```

用途：测试 FNO 在不同训练 grid resolution 下的表现。

#### Fourier/SIREN ablation

```text
results/ablate_fourier_depth3/
results/ablate_fourier_depth5/
results/ablate_fourier_depth8/

results/ablate_siren_depth3/
results/ablate_siren_depth5/
results/ablate_siren_depth8/
```

用途：说明 Fourier/SIREN 对 depth 敏感，尤其 SIREN 深层版本训练不稳定。

#### Smoke test

```text
results/smoke_fcn/
results/smoke_fourier/
results/smoke_siren/
results/smoke_residual/
results/smoke_fno/
```

用途：只用于 debugging，不用于 final report。

## 9. 最重要的最终结果

主实验结果如下：

| Model | rel L2 u | rel L2 v | rel L2 p | lambda1 abs err | lambda2 abs err | 解释 |
|---|---:|---:|---:|---:|---:|---|
| FCN PINN | 0.0422 | 0.0574 | 0.0651 | 0.0098 | 0.0013 | 强 baseline |
| Deep FCN PINN | 0.0268 | 0.0188 | 0.0750 | 0.0145 | 0.0014 | velocity 更好，但参数恢复略弱 |
| Fourier PINN | 0.0576 | 0.0569 | 0.0749 | 0.6301 | 0.0086 | field 还行，但参数恢复弱 |
| SIREN PINN | 0.0176 | 0.0194 | 0.0348 | 0.3556 | 0.0048 | 浅层优化后 field 好，但参数恢复仍弱 |
| Residual PINN | 0.0110 | 0.0084 | 0.0301 | 0.0031 | 0.0003 | 最好的完整 inverse PINN |
| FNO | 0.0200 | 0.0204 | 0.0314 | N/A | N/A | 快速强 field predictor，不做参数反演 |

最重要结论：

```text
Residual PINN is the strongest full inverse-PINN model.
```

它相对原始 FCN PINN 有明显提升：

- velocity 总误差大约从 `0.0996` 降到 `0.0194`
- `lambda_1` error 从 `0.0098` 降到 `0.0031`
- `lambda_2` error 从 `0.0013` 降到 `0.0003`

FNO 的结论要谨慎写：

```text
FNO is strong and fast for field prediction, but it does not perform inverse coefficient recovery in our implementation.
```

Fourier 和 SIREN 的结论也很有价值：

```text
Better coordinate representation can improve field fitting, but does not automatically guarantee better physical parameter recovery.
```

## 10. 如何复现实验

### 10.1 最小测试

```powershell
cd C:\Users\zbrrrrr\Desktop\228-clean
powershell -ExecutionPolicy Bypass -File experiments\run_smoke.ps1
```

### 10.2 跑一个单独模型

例如跑 Residual PINN：

```powershell
cd C:\Users\zbrrrrr\Desktop\228-clean
$env:PYTHONPATH = "src"
conda run -n 5080py python -m pinn_project.train --model residual --run-name demo_residual --steps 3000 --n-obs 5000 --data-batch 2048 --colloc-batch 2048 --width 128 --depth 6
```

### 10.3 重新生成 summary

```powershell
$env:PYTHONPATH = "src"
conda run -n 5080py python -m pinn_project.summarize --results-dir results --out results/final_summary.csv --exclude-prefix smoke_
```

### 10.4 重新生成 figures

```powershell
$env:PYTHONPATH = "src"
conda run -n 5080py python -m pinn_project.make_figures --summary results/final_summary.csv --out-dir results/figures
```

### 10.5 重新跑完整实验

只有在改了模型代码、loss 或实验设置后才需要重新跑：

```powershell
powershell -ExecutionPolicy Bypass -File experiments\run_full_suite.ps1
```

预计会比较久，但 5080 可以跑。

## 11. 报告应该怎么写

最终报告不要写成：

```text
We trained a PINN and tuned hyperparameters.
```

这样正好会踩老师的反馈。

应该写成：

```text
We compare vanilla PINNs, specialized coordinate networks, and neural-operator-style models for inverse Navier-Stokes learning.
```

推荐贡献点：

1. We build a multi-architecture benchmark instead of only tuning one FCN.
2. We compare coordinate-based PINNs with Fourier-feature and SIREN variants.
3. We include FNO as a neural-operator-style baseline.
4. We evaluate field error, PDE residual, continuity residual, parameter recovery, robustness, extrapolation, runtime, and GPU memory.
5. We find that residual connections give the best full inverse-PINN performance.

### 11.1 Introduction 要写什么

重点：

- 普通神经网络可能预测得像，但不满足物理规律
- PINN 把 PDE residual 加进 loss
- 只做 FCN PINN 不够，所以我们比较多个结构
- 项目和课程 Week 7 Neural Operator、Week 8 PINN 对应

### 11.2 Related Work 要写什么

按主题写，不要单纯列论文：

- PINNs
- Coordinate-based neural fields
- Fourier features
- SIREN
- Neural operators / FNO / DeepONet
- PINN training difficulties

### 11.3 Method 要写什么

必须包括：

- Navier-Stokes 方程
- Taylor-Green vortex 解析解
- 输入输出定义
- Data loss
- PDE residual loss
- Continuity loss
- `lambda_1, lambda_2` 参数恢复
- 每个模型结构简介

### 11.4 Experiments 要写什么

列出：

- Architecture comparison
- Data scarcity
- Noise robustness
- Time extrapolation
- FNO resolution
- Ablation
- Metrics

### 11.5 Results 要写什么

重点讲：

- Residual PINN 最强
- FNO 快且准，但不是完整 inverse model
- Fourier/SIREN 有表示优势，但物理一致性/参数恢复不一定好
- 这说明 physical ML 不能只看 supervised field error

### 11.6 Conclusion 要写什么

总结：

- 项目从 basic FCN regression 升级为 architecture comparison
- residual architecture 对 inverse PINN 很有帮助
- neural operator 是强 field baseline
- 未来可以做 physics-informed FNO、DeepONet、cylinder wake、multi-seed

## 12. Presentation 应该怎么讲

3 分钟很短，不要讲太多细节。

建议结构：

1. 30 秒：问题和老师反馈
2. 40 秒：我们比较了哪些模型
3. 40 秒：实验设置和 metrics
4. 60 秒：主结果图和结论
5. 20 秒：takeaway 和 limitation

最适合展示的图：

```text
results/figures/architecture_summary.png
```

一句话 takeaway：

```text
Residual PINN gives the best physics-informed inverse result, while FNO is a fast and accurate field-prediction baseline.
```

## 13. 队友分工建议

可以这样分：

### Person 1: 实验和代码负责人

负责：

- 确认 `results/final_summary.csv`
- 如果需要，重跑某些实验
- 检查 metrics 是否一致
- 回答代码相关问题

重点文件：

```text
src/pinn_project/train.py
experiments/run_full_suite.ps1
results/final_summary.csv
```

### Person 2: 图表和结果负责人

负责：

- 从 `results/figures/` 选图
- 从 `final_summary.csv` 做表格
- 检查图中文字是否清晰
- 整理 report 中的 Results section

重点文件：

```text
results/figures/
docs/final_results_summary.md
```

### Person 3: Method 和 Related Work 负责人

负责：

- 写 PINN loss
- 写 Navier-Stokes residual
- 写模型结构
- 写相关工作

重点文件：

```text
docs/final_report_draft.md
src/pinn_project/physics.py
src/pinn_project/models.py
```

### Person 4: Presentation 和最终整合负责人

负责：

- 做 3 分钟 slides
- 整合最终 report
- 检查 narrative 是否回答老师反馈
- 控制 presentation 时间

重点文件：

```text
docs/final_presentation_script.md
docs/presentation_plan.md
docs/final_report_draft.md
```

## 14. 写作时的几个重要 caveats

### 14.1 不要说 FNO 完成了 inverse parameter recovery

当前 FNO 不学习 `lambda_1, lambda_2`，所以不能和 PINN 一样讨论 lambda recovery。

正确写法：

```text
FNO is included as a neural-operator-style field prediction baseline.
```

### 14.2 不要隐藏 Fourier/SIREN 的不足

Fourier 和 SIREN 并不是所有指标都赢。但这不是坏事。

可以写成：

```text
Expressive coordinate representations can fit fields well, but physical consistency and coefficient recovery require additional care.
```

这会让分析更成熟。

### 14.3 不要说我们用了真实 cylinder wake 数据做最终主实验

我们最终主实验是 Taylor-Green vortex analytic benchmark。原始 cylinder wake 是 proposal 和 `pinns-torch` reference。

正确写法：

```text
We use Taylor-Green vortex as a controlled benchmark for reproducible architecture comparison. The original cylinder wake example remains a natural future extension.
```

### 14.4 不要把项目说成只调超参数

老师已经明确提醒过这个问题。

要强调：

- architecture comparison
- physical residual
- parameter recovery
- neural operator baseline
- robustness/generalization

## 15. 如果继续优化，可以做什么

如果还有时间，可以考虑：

1. 多 seed 重跑主实验，报告 mean/std
2. 给 FNO 加 PDE residual，做 physics-informed FNO
3. 加 DeepONet baseline
4. 用原始 `pinns-torch` cylinder wake 数据再跑一个小实验
5. 优化 Fourier PINN 的 loss balancing
6. 做更漂亮的 velocity quiver/streamline 图
7. 把 report 草稿转成 LaTeX/NeurIPS template

优先级最高的是：

```text
把 final_report_draft.md 整理成正式 6-8 页 PDF，并把 architecture_summary.png 和 slice_t050.png 插进去。
```

## 16. 最终交付 checklist

提交前确认：

- [ ] `results/final_summary.csv` 存在
- [ ] `results/figures/architecture_summary.png` 存在
- [ ] report 中明确回应老师反馈
- [ ] report 中没有把 FNO 写成 inverse parameter recovery model
- [ ] report 中解释 Taylor-Green benchmark 的原因
- [ ] GitHub README 有复现命令
- [ ] final report 有 GitHub repo link
- [ ] presentation 控制在 3 分钟内
- [ ] slides 不要塞太多文字

## 17. 最短接手路线

如果队友时间很少，只需要按这个顺序看：

1. `docs/final_results_summary.md`
2. `results/figures/architecture_summary.png`
3. `docs/final_report_draft.md`
4. `docs/final_presentation_script.md`
5. `README.md`
6. 当前 `docs/project_handoff.md`

这样就能理解项目主线、结果和怎么讲。
