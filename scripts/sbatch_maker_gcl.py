import os
from itertools import product

sbatch_template = """#!/bin/bash

# Vector
##SBATCH --account=deadline
##SBATCH --qos=deadline
##SBATCH --partition=t4v1,p100
##SBATCH --mem=16G
##SBATCH --cpus-per-task=16

# CSLab
#SBATCH --partition=biggpunodes
#SBATCH --cpus-per-task=4

#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=${job_name}
#SBATCH --output=${job_name}.out
#SBATCH --error=${job_name}.error

echo `date`: Job $SLURM_JOB_ID is allocated resource

python3 ../../src/ssl-train.py \\
  --task-name ${job_name} \\
  --dim 128 \\
  --n_rounds 26 \\
  --epochs 200000 \\
  --num_batch 1 \\
  --log_dir ../../log/ssl \\
  --model_dir ${model_dir} \\
  --min_n 3 \\
  --max_n 10 \\
  --data_source ${data_source} \\
  --nvar_step 1 \\
  --learning_rate ${learning_rate} \\
  --num_workers 4 \\
  --test_epoch 50 \\
  --label_proportion 0.10 \\
  --batch_size  ${batch_size}\\
  --weight_decay 1e-6 \\
  --val_folder ${val_folder} \\
  --val_instance ${val_instance} \\
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
  --simclr_tau  ${simclr_tau}\\
  --vicreg_lambda 15 \\
  --vicreg_mu 1 \\
  --vicreg_nu 1\\
  --world_size 1 \\
  --DFS \\
  --wandb_login '6ccc1788007766873b19ce42654ba2242d8b7588' \\
  --wandb_proj 'rep4co' \\
  --wandb_entity 'capitan_haddock' \\
  #  --debug
#  --reverse_aug 
  """


# SR(3, 10)
model_dir = "../../model/sr3to10_ssl"
val_folder = "../../data/val_sr_3_10_1000"
val_instance = "sat_instance_"
data_source = "sr"
job_name_suffix = "s3"

# Power
# model_dir = "../../model/power"
# val_folder = "../../data/power_10_1000"
# val_instance = "test_"
# data_source = "power"
# job_name_suffix = "power"

# DoublePower
# model_dir = "../../model/double_power"
# val_folder = "../../data/double_power"
# val_instance = "test_"
# data_source = "double_power"
# job_name_suffix = "double_power"


def set_defaults(sbatch_command, lr=0.0002, batch_size=128, tau=0.5):
    sbatch_command = sbatch_command.replace("${learning_rate}", str(lr))
    sbatch_command = sbatch_command.replace("${batch_size}", str(batch_size))
    sbatch_command = sbatch_command.replace("${simclr_tau}", str(tau))

    return sbatch_command


def sweep_ratios():
    ranges = {
        "del_clause_ratio": [0.01, 0.05, 0.10, 0.15, 0.20],
        "del_var_ratio": [0.01, 0.05, 0.10, 0.15, 0.20],
        "purt_link_ratio": [0.01, 0.05, 0.10, 0.15, 0.20],
        "subgraph_ratio": [0.01, 0.05, 0.10, 0.15, 0.20]
    }

    os.makedirs("gcl_scripts", exist_ok=True)

    for purt_type, purt_range in ranges.items():
        for ratio in purt_range:
            sbatch_command = set_defaults(f"{sbatch_template}")
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

            sbatch_command = sbatch_command.replace("${model_dir}", model_dir)
            sbatch_command = sbatch_command.replace("${val_instance}", val_instance)
            sbatch_command = sbatch_command.replace("${val_folder}", val_folder)
            sbatch_command = sbatch_command.replace("${data_source}", data_source)
            sbatch_command = sbatch_command.replace("${job_name_suffix}", job_name_suffix)

            job_name = f"neurosat_gcl_{purt_type}_{ratio}_{job_name_suffix}_heavy_diffaug"
            sbatch_command = sbatch_command.replace("${job_name}", job_name)

            print(f"Writing {job_name}")
            script_file = "gcl_scripts/{}.sh".format(job_name)
            with open(script_file, "w") as f:
                f.write(sbatch_command)

    print("Done!")

# TODO: Add the reverse_aug
def make_heatmap():
    ranges = {
        "del_clause_ratio": 0.05,
        "del_var_ratio": 0.15,
        "purt_link_ratio": 0.20,
        "subgraph_ratio": 0.01
    }

    os.makedirs("gcl_scripts", exist_ok=True)

    for type_1 in ranges:
        for type_2 in ranges:
            if type_1 <= type_2:
                continue

            sbatch_command = set_defaults(f"{sbatch_template}")

            purt_types = [type_1, type_2]
            del_cla = int("del_clause_ratio" in purt_types)
            del_var = int("del_var_ratio" in purt_types)
            del_link = int("purt_link_ratio" in purt_types)
            del_subgraph = int("subgraph_ratio" in purt_types)

            sbatch_command = sbatch_command.replace("${gcl_clause_drop}", str(del_cla))
            sbatch_command = sbatch_command.replace("${gcl_var_drop}", str(del_var))
            sbatch_command = sbatch_command.replace("${gcl_link_purt}", str(del_link))
            sbatch_command = sbatch_command.replace("${gcl_subgraph}", str(del_subgraph))
            sbatch_command = sbatch_command.replace("${gcl_del_clause_ratio}", str(del_cla * ranges["del_clause_ratio"]))
            sbatch_command = sbatch_command.replace("${gcl_del_var_ratio}", str(del_var * ranges["del_var_ratio"]))
            sbatch_command = sbatch_command.replace("${gcl_purt_link_ratio}", str(del_link * ranges["purt_link_ratio"]))
            sbatch_command = sbatch_command.replace("${gcl_subgraph_ratio}", str(del_subgraph * ranges["subgraph_ratio"]))

            sbatch_command = sbatch_command.replace("${model_dir}", model_dir)
            sbatch_command = sbatch_command.replace("${val_instance}", val_instance)
            sbatch_command = sbatch_command.replace("${val_folder}", val_folder)
            sbatch_command = sbatch_command.replace("${data_source}", data_source)
            sbatch_command = sbatch_command.replace("${job_name_suffix}", job_name_suffix)

            job_name = f"neurosat_gcl_heat_{type_1}_{ranges[type_1]}_{type_2}_{ranges[type_2]}_{job_name_suffix}_heavy_diffaug"
            sbatch_command = sbatch_command.replace("${job_name}", job_name)

            print(f"Writing {job_name}")
            script_file = "gcl_scripts/{}.sh".format(job_name)
            with open(script_file, "w") as f:
                f.write(sbatch_command)


def sweep_hypers():
    ranges = {
        "del_clause_ratio": 0.05,
        "del_var_ratio": 0.0,
        "purt_link_ratio": 0.0,
        "subgraph_ratio": 0.0
    }

    hyper_ranges = {
        "learning_rate": [2e-3, 2e-4, 2e-5],
        "batch_size": [128, 256],
        "simclr_tau": [0.01, 0.05, 0.1, 0.5, 1]
    }

    os.makedirs("gcl_scripts", exist_ok=True)

    for it, hyper in enumerate(product(*hyper_ranges.values())):
        sbatch_command = set_defaults(f"{sbatch_template}", hyper[0], hyper[1], hyper[2])

        del_cla = int(ranges['del_clause_ratio'] > 0)
        del_var = int(ranges['del_var_ratio'] > 0)
        del_link = int(ranges['purt_link_ratio'] > 0)
        del_subgraph = int(ranges['subgraph_ratio'] > 0)

        sbatch_command = sbatch_command.replace("${gcl_clause_drop}", str(del_cla))
        sbatch_command = sbatch_command.replace("${gcl_var_drop}", str(del_var))
        sbatch_command = sbatch_command.replace("${gcl_link_purt}", str(del_link))
        sbatch_command = sbatch_command.replace("${gcl_subgraph}", str(del_subgraph))
        sbatch_command = sbatch_command.replace("${gcl_del_clause_ratio}", str(ranges['del_clause_ratio']))
        sbatch_command = sbatch_command.replace("${gcl_del_var_ratio}", str(ranges['del_var_ratio']))
        sbatch_command = sbatch_command.replace("${gcl_purt_link_ratio}", str(ranges['purt_link_ratio']))
        sbatch_command = sbatch_command.replace("${gcl_subgraph_ratio}", str(ranges['subgraph_ratio']))

        sbatch_command = sbatch_command.replace("${model_dir}", model_dir)
        sbatch_command = sbatch_command.replace("${val_instance}", val_instance)
        sbatch_command = sbatch_command.replace("${val_folder}", val_folder)
        sbatch_command = sbatch_command.replace("${data_source}", data_source)
        sbatch_command = sbatch_command.replace("${job_name_suffix}", job_name_suffix)

        job_name = f"neurosat_gcl_{it}_lr_{hyper[0]}_bs_{hyper[1]}_tau_{hyper[2]}_heavy_diffaug"
        sbatch_command = sbatch_command.replace("${job_name}", job_name)

        print(f"Writing {job_name}")
        script_file = "gcl_scripts/{}.sh".format(job_name)
        with open(script_file, "w") as f:
            f.write(sbatch_command)

    print("Done!")


sweep_hypers()
# make_heatmap()
# sweep_ratios()
