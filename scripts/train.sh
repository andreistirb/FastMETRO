export PYTHONPATH=/kits/FastMETRO/:/kits/FastMETRO/manopth

python src/tools/run_fastmetro_handmesh.py \
       --train_yaml /kits/Datasets/freihand/freihand/train.yaml \
       --val_yaml /kits/Datasets/freihand/freihand/test.yaml \
       --arch hrnet-w64 \
       --model_name FastMETRO-L \
       --num_workers 4 \
       --per_gpu_train_batch_size 12 \
       --per_gpu_eval_batch_size 12 \
       --lr 1e-4 \
       --num_train_epochs 200 \
       --output_dir experiments/fast_metro_with_mano/ \
       --visualize_training \
       --visualize_multi_view \
       --logging_steps 2000 \
       --use_smpl_param_regressor

