
# learning_rates_pedi=(0.0001 0.00001 0.00005 0.0005 0.001)

# for lr in "${learning_rates_pedi[@]}"
# do
#     ./isaaclab.sh -p scripts/p4rl/rsl_rl/train.py --task P4RL-Pedipulation-Flat-Blind-Anymal-D-v0 --logger wandb --headless --seed -1 --learning_rate $lr
# done

#---------------------------------------------------------------

# learning_rates=(0.00001 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1)

# for lr in "${learning_rates[@]}"
# do
#     ./isaaclab.sh -p scripts/p4rl/rsl_rl/train.py --task P4RL-Velocity-Flat-Anymal-D-v0 --logger wandb --headless --seed -1 --learning_rate $lr
# done


#---------------------------------------------------------------

# epochs=(200 500 1000 2000 3000)

# for epochs in "${epochs[@]}"
# do 
#     ./isaaclab.sh -p ./scripts/p4rl/rsl_rl/train.py --task P4RL-Pedipulation-Flat-Blind-Anymal-D-v0 --headless --seed 0 --freeze_pretrain_until_iteration $epochs --pretrained_weights_path ./logs/rsl_rl/pretrained_weights/CDAC-loco-2.0.pt --logger wandb
# done

#---------------------------------------------------------------



./isaaclab.sh -p scripts/p4rl/rsl_rl/train.py --task P4RL-Pedipulation-Flat-Blind-Anymal-D-v0 --logger wandb --headless --seed -1

./isaaclab.sh -p scripts/p4rl/rsl_rl/train.py --task P4RL-Pre-Kinematic-Pedipulation-Flat-Blind-Anymal-D-v0 --logger wandb --headless --seed -1


# ./source/p4rl/shell/sweep_lr.sh

