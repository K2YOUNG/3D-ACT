CUDA_VISIBLE_DEVICES=1 python3 act/imitate_episodes.py \
--task_name real_pick_and_place \
--ckpt_dir ./output_batch8 \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 256 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-5 \
--seed 0 \
--num_points 260000 \
--use_pointcloud