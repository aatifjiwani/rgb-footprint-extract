#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=inference
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanjo@law.stanford.edu
#SBATCH --partition=owners
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --array=0-1

cd ../../

backbone_list="drn_c42 drn_c42
"
out_stride_list="8 8
"
workers_list="2 2
"
epochs_list="1 1
"
test_batch_size_list="1 1
"
gpu_ids_list="0 0
"
resume_list="SJ_0.2_True_0.0001_0.0001_1.03_superresx2 SJ_0.2_True_0.0001_0.0001_1.03_superresx2
"
window_size_list="256 256
"
stride_list="256 256
"
input_filename_list="/oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/train/images/m_3712141_ne_10_060_20200525_418.npy /oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superresx2/train/images/m_3712133_sw_10_060_20200525_238.npy
"
output_filename_list="/oak/stanford/groups/deho/building_compliance/rgb-footprint-extract/small_mIOU_testing/m_3712141_ne_10_060_20200525_418.npy /oak/stanford/groups/deho/building_compliance/rgb-footprint-extract/small_mIOU_testing/m_3712133_sw_10_060_20200525_238.npy
"
backbone_list=($backbone_list)
out_stride_list=($out_stride_list)
workers_list=($workers_list)
epochs_list=($epochs_list)
test_batch_size_list=($test_batch_size_list)
gpu_ids_list=($gpu_ids_list)
resume_list=($resume_list)
window_size_list=($window_size_list)
stride_list=($stride_list)
input_filename_list=($input_filename_list)
output_filename_list=($output_filename_list)


singularity exec --nv $GROUP_HOME/singularity/rgb-building1.sif python run_deeplab.py --inference --best-miou --backbone ${backbone_list[$SLURM_ARRAY_TASK_ID]} --out-stride ${out_stride_list[$SLURM_ARRAY_TASK_ID]} --workers ${workers_list[$SLURM_ARRAY_TASK_ID]} --epochs ${epochs_list[$SLURM_ARRAY_TASK_ID]} --gpu-ids ${gpu_ids_list[$SLURM_ARRAY_TASK_ID]} --resume ${resume_list[$SLURM_ARRAY_TASK_ID]} --test-batch-size ${test_batch_size_list[$SLURM_ARRAY_TASK_ID]} --window-size ${window_size_list[$SLURM_ARRAY_TASK_ID]} --stride ${stride_list[$SLURM_ARRAY_TASK_ID]} --input-filename ${input_filename_list[$SLURM_ARRAY_TASK_ID]} --output-filename ${output_filename_list[$SLURM_ARRAY_TASK_ID]}
