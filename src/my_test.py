import os

sbatch_template = """#!/bin/bash

#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --partition=t4v1,p100

#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=${job_name}
#SBATCH --output=${job_name}.out
#SBATCH --error=${job_name}.error

echo `date`: Job $SLURM_JOB_ID is allocated resource

python3 ../src/ssl-train.py \\
  --task-name ${job_name} \\
  --dim 128 \\
  --n_rounds 26 \\
  --epochs 200000 \\
  --num_batch 1 \\
  --log_dir ../log/ssl \\
  --model_dir ../model/sr3to10_ssl \\
  --min_n 3 \\
  --max_n 10 \\
  --data_source sr \\
  --nvar_step 1 \\
  --learning_rate 0.0002 \\
  --num_workers 8 \\
  --test_epoch 20 \\
  --label_proportion 0.10 \\
  --batch_size 80 \\
  --weight_decay 1e-6 \\
  --val_folder ../data/val_sr_3_10_1000 \\
  --val_instance sat_instance \\
  --val_label label.pkl \\
  --val_n_vars n_vars.pkl \\
  --variable_elimination 0 \\
  --clause_resolution 0 \\
  --subsume_clause 0 \\
  --add_trivial 0 \\
  --blocked_clause 0 \\
  --gcl_clause_drop ${gcl_clause_drop} \\
  --gcl_var_drop ${gcl_var_drop} \\
  --gcl_link_purt ${gcl_link_purt} \\
  --gcl_subgraph ${gcl_subgraph} \\
  --at_added_literal 0.0 \\
  --at_added_clause 0.0 \\
  --cr_added_resolv 0.0 \\
  --ve_eliminate_var 0.0 \\
  --ve_max_resolvent 0.0 \\
  --gcl_del_clause_ratio ${gcl_del_clause_ratio} \\
  --gcl_del_var_ratio ${gcl_del_var_ratio} \\
  --gcl_purt_link_ratio ${gcl_purt_link_ratio} \\
  --gcl_subgraph_ratio ${gcl_subgraph_ratio} \\
  --save_model \\
  --neurosat \\
  --num_ve 0.1 \\
  --simclr \\
  --ssl \\
  --reverse_aug \\
  --simclr_tau 0.5 \\
  --vicreg_lambda 15 \\
  --vicreg_mu 1 \\
  --vicreg_nu 1\\
  --world_size 1 \\
  --DFS \\
  #  --debug \\"""

ranges = {
    "del_clause_ratio": [0.00, 0.05, 0.10, 0.15, 0.20],
    "del_var_ratio": [0.00, 0.05, 0.10, 0.15, 0.20],
    "purt_link_ratio": [0.00, 0.05, 0.10, 0.15, 0.20],
    "subgraph_ratio": [0.00, 0.05, 0.10, 0.15, 0.20],
}


os.makedirs("gcl_scripts", exist_ok=True)

for purt_type, purt_range in ranges.items():
    for ratio in purt_range:
        sbatch_command = f"{sbatch_template}"
        del_cla = int(purt_type == "del_clause_ratio")
        del_var = int(purt_type == "del_var_ratio")
        del_link = int(purt_type == "purt_link_ratio")
        del_subgraph = int(purt_type == "subgraph_ratio")

        sbatch_command = sbatch_command.replace("${gcl_clause_drop}", str(del_cla))
        sbatch_command = sbatch_command.replace("${gcl_var_drop}", str(del_var))
        sbatch_command = sbatch_command.replace("${gcl_link_purt}", str(del_link))
        sbatch_command = sbatch_command.replace("${gcl_subgraph}", str(del_subgraph))
        sbatch_command = sbatch_command.replace("${gcl_del_clause_ratio}", str(del_cla * ratio))
        sbatch_command = sbatch_command.replace("${gcl_del_var_ratio}", str(del_var * ratio))
        sbatch_command = sbatch_command.replace("${gcl_purt_link_ratio}", str(del_link * ratio))
        sbatch_command = sbatch_command.replace("${gcl_subgraph_ratio}", str(del_subgraph * ratio))

        job_name = f"neurosat_gcl_{purt_type}_{ratio}_s3_heavy_diffaug"
        sbatch_command = sbatch_command.replace("${job_name}", job_name)

        print(f"Writing {job_name}")
        script_file = "gcl_scripts/{}.sh".format(job_name)
        with open(script_file, "w") as f:
            f.write(sbatch_command)

print("Done!")