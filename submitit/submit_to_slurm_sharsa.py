import os
from pathlib import Path
from datetime import datetime
import platform

import submitit


def main():
    cluster_name = platform.node().split('-')[0]
    if cluster_name == 'adroit':
        log_root_dir = '/home/cz8792/network'
        partition = 'gpu'
        account = None
        exclude = None
    elif cluster_name == 'della':
        log_root_dir = '/home/cz8792/gpfs'
        partition = 'gpu-test'
        account = None
        exclude = None
    elif cluster_name in ['soak.cs.princeton.edu', 'wash.cs.princeton.edu',
                          'rinse.cs.princeton.edu', 'spin.cs.princeton.edu']:
        log_root_dir = '/n/fs/rl-chongyiz'
        partition = None
        account = 'pnlp'
        exclude = None
    elif cluster_name == 'neuronic.cs.princeton.edu':
        log_root_dir = '/n/fs/prl-chongyiz'
        partition = 'all'
        account = None
        exclude = 'neu301'
    else:
        raise NotImplementedError

    executor = submitit.AutoExecutor(folder="/tmp/submitit_logs")  # this path is not actually used.
    executor.update_parameters(
        slurm_name="sharsa",
        slurm_time=int(48 * 60),  # minute
        slurm_partition=partition,
        slurm_account=account,
        slurm_nodes=1,
        slurm_ntasks_per_node=1,  # tasks can share nodes
        slurm_cpus_per_task=8,
        slurm_mem="16G",
        slurm_gpus_per_node=1,
        slurm_exclude=exclude,
        slurm_stderr_to_stdout=True,
        slurm_array_parallelism=10,
    )

    with executor.batch():  # job array
        for env_name in ["cube-octuple-play-oraclerep-v0"]:
            for dataset_name in ["cube-octuple-play-100m-v0"]:
                for seed in [1, 2, 3]:
                    exp_name = f"{datetime.today().strftime('%Y%m%d')}_sharsa_{env_name}_{dataset_name}"
                    log_dir = os.path.expanduser(
                        f"{log_root_dir}/exp_logs/horizon_reduction_logs/sharsa/{exp_name}/{seed}")

                    # change the log folder of slurm executor
                    submitit_log_dir = os.path.join(os.path.dirname(log_dir),
                                                    'submitit')
                    executor._executor.folder = Path(
                        submitit_log_dir).expanduser().absolute()

                    cmds = f"""
                        unset PYTHONPATH;
                        source $HOME/.zshrc;
                        conda activate horizon-reduction;
                        which python;
                        echo $CONDA_PREFIX;

                        echo job_id: $SLURM_ARRAY_JOB_ID;
                        echo task_id: $SLURM_ARRAY_TASK_ID;
                        squeue -j $SLURM_JOB_ID -o "%.18i %.9P %.8j %.8u %.2t %.6D %.5C %.11m %.11l %.12N";
                        echo seed: {seed};

                        export PROJECT_DIR=$PWD;
                        export PYTHONPATH=$PROJECT_DIR;
                        export PATH="$PATH":"$CONDA_PREFIX"/bin;
                        export CUDA_VISIBLE_DEVICES=0;
                        export MUJOCO_GL=egl;
                        export PYOPENGL_PLATFORM=egl;
                        export EGL_DEVICE_ID=0;
                        source $HOME/env_vars.sh;
                        export D4RL_SUPPRESS_IMPORT_ERROR=1;
                        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia;
                        export XLA_FLAGS=--xla_gpu_triton_gemm_any=true;

                        rm -rf {log_dir};
                        mkdir -p {log_dir};
                        python $PROJECT_DIR/main.py \
                            --enable_wandb=1 \
                            --env_name={env_name} \
                            --dataset_dir={log_root_dir}/datasets/horizon-reduction/{dataset_name} \
                            --dataset_replace_interval=1000 \
                            --agent=agents/sharsa.py \
                            --agent.q_agg=min \
                            --agent.subgoal_steps=50 \
                            --agent.actor_p_trajgoal=1.0 \
                            --agent.actor_p_randomgoal=0.0 \
                            --agent.actor_geom_sample=True \
                            --seed={seed} \
                            --save_dir={log_dir} \
                        2>&1 | tee {log_dir}/stream.log;

                        export SUBMITIT_RECORD_FILENAME={log_dir}/submitit_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID".txt;
                        echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_submitted.pkl" >> "$SUBMITIT_RECORD_FILENAME";
                        echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_submission.sh" >> "$SUBMITIT_RECORD_FILENAME";
                        echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_0_log.out" >> "$SUBMITIT_RECORD_FILENAME";
                        echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_0_result.pkl" >> "$SUBMITIT_RECORD_FILENAME";
                    """

                    cmd_func = submitit.helpers.CommandFunction([
                        "/bin/zsh", "-c",
                        cmds,
                    ], verbose=True)

                    executor.submit(cmd_func)


if __name__ == "__main__":
    main()
