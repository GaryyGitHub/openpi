# OpenPi Aloha-Sim 复现流程总结

## 1. 项目简介
OpenPi 是一个机器人仿真平台，旨在通过高效的仿真环境支持机器人任务的开发与评估。`aloha-sim` 是其中一个仿真环境，主要用于模拟 ALOHA 机器人完成抓取、搬运等任务。

- **核心特点**：
  - 提供高效的仿真环境，支持多种任务类型。
  - 通过策略服务与仿真环境交互，实现动作控制。
  - 支持多回合评估，便于分析策略性能。

- **任务目标**：完成指定任务（如 `AlohaTransferCube-v0`），并通过多次评估计算成功率。

---

## 2. 环境配置

### 2.1 系统依赖
- **操作系统**：建议使用 Linux（如 Ubuntu 20.04）。
- **必要依赖**：
  ```bash
  sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
  ```

### 2.2 Python 环境
1. 创建虚拟环境：
   ```bash
   uv venv --python 3.10 examples/aloha_sim/.venv
   source examples/aloha_sim/.venv/bin/activate
   ```
2. 安装依赖：
   ```bash
   uv pip sync examples/aloha_sim/requirements.txt
   uv pip install -e packages/openpi-client
   ```

---

## 3. 运行流程

### 3.1 启动策略服务
在一个终端中运行：
```bash
uv run scripts/serve_policy.py --env ALOHA_SIM
```
- **作用**：加载训练好的策略模型，启动 WebSocket 服务（默认端口 8000）。
- **注意**：服务需保持运行，供仿真环境调用。

### 3.2 启动仿真环境
在另一个终端中运行：
```bash
MUJOCO_GL=egl python examples/aloha_sim/main.py --num-episodes 20 --success-reward-threshold 0.5
```
- **作用**：
  - 启动仿真环境并连接策略服务。
  - 执行任务并记录每个回合的得分与成功率。
- **参数说明**：
  - `--num-episodes`：运行的回合数量。
  - `--success-reward-threshold`：成功判定的奖励阈值。

---

## 4. 值得注意的细节

1. **EGL 渲染**：
   - 如果遇到 EGL 错误，确保安装了 `libegl1-mesa-dev` 和 `libgles2-mesa-dev`。
   - 使用 `MUJOCO_GL=egl` 环境变量强制启用 EGL 渲染。

2. **成功判定**：
   - 默认奖励阈值为 `0.5`，可通过 `--success-reward-threshold` 调整。
   - 如果任务奖励稀疏（如成功=1，失败=0），建议设置为 `1.0`。

3. **多回合评估**：
   - 使用 `--num-episodes` 控制评估次数。
   - 每个回合的得分和成功状态会实时打印。

4. **步数限制**：
   - 可通过 `--max-episode-steps` 设置每个回合的最大步数。
   - 默认无限制，由环境内置逻辑决定。

---

## 5. 示例输出
运行仿真后，终端会输出类似：
```
Episode 1: Score=1.0000 ✓ Success | Current Success Rate: 1/1 (100.00%)
Episode 2: Score=0.0000 ✗ Failed | Current Success Rate: 1/2 (50.00%)
...
Success: 13/20 (65.00%)
```

---

## 6. 总结
通过以上流程，可以成功复现 OpenPi 的 `aloha-sim` 仿真任务。用户可以根据需求调整任务类型、评估参数等，进一步分析策略性能。