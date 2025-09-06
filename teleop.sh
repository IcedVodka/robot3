#!/bin/bash

# 显示菜单
echo "=========================================="
echo "        机器人遥操作任务选择菜单"
echo "=========================================="
echo "1. 从柜子里抓取杯子"
echo "2. 举起杯子并调整姿势"
echo "3. 将杯子放在碟子上"
echo "4. 抓取蓝色方块"
echo "5. 将蓝色方块放入杯子中"
echo "=========================================="

# 读取用户输入
read -p "请选择任务 (1-5): " choice

# 根据选择执行对应命令
case $choice in
    1)
        echo "执行任务1: 从柜子里抓取杯子"
        python -m examples.teleop \
            --master-ip 192.168.1.18 \
            --slave-ip 192.168.1.19 \
            --hand \
            --task-name "task2_5_1" \
            --task-description "Grasp the cup from the cabinet." \
            --gripper-init 0
        ;;
    2)
        echo "执行任务2: 举起杯子并调整姿势"
        python -m examples.teleop \
            --master-ip 192.168.1.18 \
            --slave-ip 192.168.1.19 \
            --hand \
            --task-name "task2_5_2" \
            --task-description "Lift the cup and adjust your posture." \
            --gripper-init 1
        ;;
    3)
        echo "执行任务3: 将杯子放在碟子上"
        python -m examples.teleop \
            --master-ip 192.168.1.18 \
            --slave-ip 192.168.1.19 \
            --hand \
            --task-name "task2_5_3" \
            --task-description "Place the cup on the saucer." \
            --gripper-init 1
        ;;
    4)
        echo "执行任务4: 抓取蓝色方块"
        python -m examples.teleop \
            --master-ip 192.168.1.18 \
            --slave-ip 192.168.1.19 \
            --hand \
            --task-name "task2_5_4" \
            --task-description "Grasp the blue block." \
            --gripper-init 0
        ;;
    5)
        echo "执行任务5: 将蓝色方块放入杯子中"
        python -m examples.teleop \
            --master-ip 192.168.1.18 \
            --slave-ip 192.168.1.19 \
            --hand \
            --task-name "task2_5_5" \
            --task-description "Place the blue block in the cup." \
            --gripper-init 1
        ;;
    *)
        echo "无效选择！请输入1-5之间的数字。"
        exit 1
        ;;
esac

echo "任务执行完成！"