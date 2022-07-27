import os
import sys

def put_qmark(s):
    s = "\"" + s +"\""
    return s


def generate(backbone, out_stride, workers, epochs, test_batch_size, gpu_ids, resume, window_size, stride, input_filename, output_filename, array):

    backbone_list = []
    out_stride_list = []
    workers_list = []
    epochs_list = []
    test_batch_size_list = []
    gpu_ids_list = []
    resume_list = []
    window_size_list = []
    stride_list = []
    input_filename_list = []
    output_filename_list = []
    for b in backbone:
        for o in out_stride:
            for w in workers:
                for e in epochs:
                    for tb in test_batch_size:
                        for g in gpu_ids:
                            for r in resume:
                                for wi in window_size:
                                    for s in stride:
                                        for i in input_filename:
                                            for of in output_filename:
                                                backbone_list.append(b)
                                                out_stride_list.append(o)
                                                workers_list.append(w)
                                                epochs_list.append(e)
                                                test_batch_size_list.append(tb)
                                                gpu_ids_list.append(g)
                                                resume_list.append(r)
                                                window_size_list.append(wi)
                                                stride_list.append(s)
                                                input_filename_list.append(i)
                                                output_filename_list.append(of)



    S = "#!/bin/bash\n"
    S += "#SBATCH --begin=now\n"
    S += "#SBATCH --job-name=inference\n"
    S += "#SBATCH --mail-type=ALL\n"
    S += "#SBATCH --mail-user=nathanjo@law.stanford.edu\n"
    S += "#SBATCH --partition=owners\n"
    S += "#SBATCH --mem=16GB\n"
    S += "#SBATCH --gres=gpu:1\n"
    S += "#SBATCH --time=00:05:00\n"
    S += "#SBATCH --array=0-"
    S += str(array)

    S += "\n"
    S += "\n"

    S += "cd ../../"

    S += "\n"
    S += "\n"

    S += "backbone_list=" + put_qmark(" ".join(str(item) for item in backbone_list) + "\n")
    S += "\n"
    S += "out_stride_list=" + put_qmark(" ".join(str(item) for item in out_stride_list) + "\n")
    S += "\n"
    S += "workers_list=" + put_qmark(" ".join(str(item) for item in workers_list) + "\n")
    S += "\n"
    S += "epochs_list=" + put_qmark(" ".join(str(item) for item in epochs_list) + "\n")
    S += "\n"
    S += "test_batch_size_list=" + put_qmark(" ".join(str(item) for item in test_batch_size_list) + "\n")
    S += "\n"
    S += "gpu_ids_list=" + put_qmark(" ".join(str(item) for item in gpu_ids_list) + "\n")
    S += "\n"
    S += "resume_list=" + put_qmark(" ".join(str(item) for item in resume_list) + "\n")
    S += "\n"
    S += "window_size_list=" + put_qmark(" ".join(str(item) for item in window_size_list) + "\n")
    S += "\n"
    S += "stride_list=" + put_qmark(" ".join(str(item) for item in stride_list) + "\n")
    S += "\n"
    S += "input_filename_list=" + put_qmark(" ".join(str(item) for item in input_filename_list) + "\n")
    S += "\n"
    S += "output_filename_list=" + put_qmark(" ".join(str(item) for item in output_filename_list) + "\n")
    S += "\n"
    S += 'backbone_list=($backbone_list)' + "\n"
    S += 'out_stride_list=($out_stride_list)' + "\n"
    S += 'workers_list=($workers_list)' + "\n"
    S += 'epochs_list=($epochs_list)' + "\n"
    S += 'test_batch_size_list=($test_batch_size_list)' + "\n"
    S += 'gpu_ids_list=($gpu_ids_list)' + "\n"
    S += 'resume_list=($resume_list)' + "\n"
    S += 'window_size_list=($window_size_list)' + "\n"
    S += 'stride_list=($stride_list)' + "\n"
    S += 'input_filename_list=($input_filename_list)' + "\n"
    S += 'output_filename_list=($output_filename_list)' + "\n"

    S += "\n"
    S += "\n"
    command = 'singularity exec --nv $GROUP_HOME/singularity/rgb-building1.sif python run_deeplab.py --inference --backbone ' + \
              '${backbone_list[$SLURM_ARRAY_TASK_ID]}' + ' --out-stride ' + '${out_stride_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --workers ' + '${workers_list[$SLURM_ARRAY_TASK_ID]}' + ' --epochs ' + '${epochs_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --gpu-ids ' + '${gpu_ids_list[$SLURM_ARRAY_TASK_ID]}'  + \
              ' --resume ' + '${resume_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --test-batch-size ' + '${test_batch_size_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --window-size ' + '${window_size_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --stride ' + '${stride_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --input-filename ' + '${input_filename_list[$SLURM_ARRAY_TASK_ID]}' + \
              ' --output-filename ' + '${output_filename_list[$SLURM_ARRAY_TASK_ID]}'

    S += command
    S += "\n"

    slurm_file = 'infer_smallmiou.sh'
    f = open(slurm_file, "w+")
    f.write(S)
    f.close()
    # print(slurm_file)


def main():

    backbone = ['drn_c42']
    out_stride = [8]
    workers = [2]
    epochs = [1]
    test_batch_size = [1]
    gpu_ids = [0]
    resume = ['SJ_0.2_True_0.0001_0.0001_1.03_superresx2']
    window_size = [256]
    stride = [256]
    input_fp = '/oak/stanford/groups/deho/building_compliance/san_jose_naip_512/phase2_superres/train/images/'
    input_filename = ['m_3712150_ne_10_060_20200525_165.npy', 'm_3712141_se_10_060_20200525_126.npy', 'm_3712142_ne_10_060_20200525_63.npy',
                    'm_3712141_ne_10_060_20200525_419.npy', 'm_3712141_se_10_060_20200525_106.npy', 'm_3712142_nw_10_060_20200525_282.npy',
                    'm_3712141_se_10_060_20200525_109.npy', 'm_3712141_ne_10_060_20200525_418.npy', 'm_3712150_nw_10_060_20200525_222.npy',
                    'm_3712141_sw_10_060_20200525_9.npy', 'm_3712141_ne_10_060_20200525_155.npy', 'm_3712133_sw_10_060_20200525_238.npy',
                    'm_3712142_sw_10_060_20200525_410.npy', 'm_3712142_sw_10_060_20200525_451.npy', 'm_3712142_sw_10_060_20200525_173.npy',
                    'm_3712141_ne_10_060_20200525_375.npy']

    output_fp = '/oak/stanford/groups/deho/building_compliance/rgb-footprint-extract/small_mIOU_testing/'
    output_filename = input_filename

    input_filename = [input_fp+i for i in input_filename]
    output_filename = [output_fp+i for i in output_filename]

    array = len(input_filename) - 1

    generate(backbone, out_stride, workers, epochs, test_batch_size, gpu_ids, resume, window_size, stride, input_filename, output_filename, array)

if __name__ == "__main__":
    main()