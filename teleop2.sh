#!/bin/bash

# 显示菜单
echo "=========================================="
echo "        机器人遥操作任务选择菜单"
echo "=========================================="
echo "1. 抓取红色方块 (Grasp the red block)"
echo "2. 将红色方块放在红色盘子上 (Place the red block on the red plate)"
echo "3. 抓取绿色方块 (Grasp the green block)"
echo "4. 将绿色方块放在绿色盘子上 (Place the green block on the green plate)"
echo "=========================================="

# 读取用户输入
read -p "请选择任务 (1-4): " choice

# 根据选择执行对应命令
case $choice in
    1)
        echo "执行任务1: 抓取红色方块 (Grasp the red block)"
        python -m examples.teleop \
            --master-ip 192.168.1.18 \
            --slave-ip 192.168.1.19 \
            --hand \
            --task-name "task1_5_1" \
            --task-description "Grasp the red block." \
            --gripper-init 0
        ;;
    2)
        echo "执行任务2: 将红色方块放在红色盘子上 (Place the red block on the red plate)"
        python -m examples.teleop \
            --master-ip 192.168.1.18 \
            --slave-ip 192.168.1.19 \
            --hand \
            --task-name "task1_5_2" \
            --task-description "Place the red block on the red plate." \
            --gripper-init 1
        ;;
    3)
        echo "执行任务3: 抓取绿色方块 (Grasp the green block)"
        python -m examples.teleop \
            --master-ip 192.168.1.18 \
            --slave-ip 192.168.1.19 \
            --hand \
            --task-name "task1_5_3" \
            --task-description "Grasp the green block." \
            --gripper-init 0
        ;;
    4)
        echo "执行任务4: 将绿色方块放在绿色盘子上 (Place the green block on the green plate)"
        python -m examples.teleop \
            --master-ip 192.168.1.18 \
            --slave-ip 192.168.1.19 \
            --hand \
            --task-name "task1_5_4" \
            --task-description "Place the green block on the green plate." \
            --gripper-init 1
        ;;
    *)
        echo "无效选择！请输入1-4之间的数字。"
        exit 1
        ;;
esac

echo "任务执行完成！"