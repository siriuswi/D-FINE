- train

CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml --use-amp --seed=0

- test
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml --test-only -r output/dfine_hgnetv2_l_custom/best_stg1.pth

- inference(visualization)
python tools/inference/torch_inf.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml -r output/dfine_hgnetv2_l_custom/best_stg1.pth --input data/obj_25/images/val/000011.jpg --device cuda:0

- tuning
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml --use-amp --seed=0 -t weight/obj365_pretrained/dfine_l_obj365.pth