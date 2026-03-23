#!/bin/bash

echo "请选择要运行的任务："
echo "0) bddl修改"
echo "1) 终端1：libero模拟"
echo "2) 终端2：模型服务器（官方ckpt）"
echo "3) 终端2：模型服务器（自定义ckpt）"
echo "4) 数据集格式转换（RLDS -> lerobot）"
echo "5) 计算归一化统计信息"
echo "6) nohup微调"
read -p "请输入数字: " choice

case $choice in
  0)
    echo ">>> bddl修改"
    uv run scripts/build_libero_multiperturbation.py   --perturbation-types position object goal   --suites libero_spatial libero_object libero_goal libero_10   --log-progress-every 5   --log-verbose
    ;;
  1)
    echo ">>> 终端1：libero模拟"
    # python examples/libero/main.py --task-suite-name libero_spatial --perturbation-type goal --perturbation-variant change_target
    # python examples/libero/main.py --task-suite-name libero_object --perturbation-type goal --perturbation-variant change_target
    # python examples/libero/main.py --task-suite-name libero_goal --perturbation-type goal --perturbation-variant change_target
    python examples/libero/main.py --task-suite-name libero_10 --perturbation-type goal --perturbation-variant change_target
    ;;
  2)
    echo ">>> 终端2：模型服务器（官方ckpt）"
    uv run scripts/serve_policy.py --env LIBERO
    ;;
  3)
    echo ">>> 终端2：模型服务器（自定义ckpt）"
    uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=/data/gaoming/openpi/checkpoints/pi05_libero/libero_lora_2/29999
    ;;
  4)
    echo ">>> 数据集格式转换（RLDS -> lerobot）..."
    uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir ./libero_data/
    ;;
  5)
    echo ">>> 计算归一化统计信息..."
    uv run scripts/compute_norm_stats.py --config-name pi05_libero
    ;;
  6)
    echo ">>> nohup微调..."
    nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_libero --exp-name=libero_lora_2 --overwrite' > train.log 2>&1 &
    ;;
  *)
    echo "无效输入。"
    exit 1
    ;;
esac