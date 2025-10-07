import getpass
import json
import subprocess
import argparse
import textwrap
import kubernetes_models
import minai
from minai import models
from pathlib import Path
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-14B-Instruct")
parser.add_argument("--max-steps", type=int, default=-1)
parser.add_argument("--n-gpus", type=int, default=1)
parser.add_argument("--n-nodes", type=int, default=1)
parser.add_argument("--training-type", type=str, default="grpo")
parser.add_argument("--workflow-name", type=str, default="normal")
args = parser.parse_args()

start_time = str(datetime.datetime.now()).split(".")[0].replace(" ", "_").replace(":", "-")
current_dir = Path.cwd()
templates: list[models.Template] = []

accelerate_env_var = textwrap.dedent(
    f"""-x ACCELERATE_USE_DEEPSPEED=true \\
    -x ACCELERATE_MIXED_PRECISION=bf16 \\
    -x ACCELERATE_DEEPSPEED_ZERO_STAGE=3 \\
    -x ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE=cpu \\
    -x ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE=cpu \\
    -x ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1 \\
    -x ACCELERATE_GRADIENT_CLIPPING=1.0 \\
    -x ACCELERATE_DEEPSPEED_ZERO3_INIT=true \\
    -x ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL=true \\
    """
)

debug_var = textwrap.dedent(
    f"""-x NCCL_DEBUG=INFO \\
-x TORCH_DISTRIBUTED_DEBUG=DETAIL \\
    """
)

mpi_job = models.MPIJob(
    spec=models.MPIJobSpec(
        launcher_template=models.PodTemplateSpec(
            spec=models.PodSpec(
                containers=[
                    models.Container(
                        name="mpi",
                        image="asia-northeast1-docker.pkg.dev/pfn-artifactregistry/all-in-one/stable@sha256:957c8e229c630ab5e90efbdbd79da3577e7b413957dd934cb670c7d3e2951114",
                        command=[
                            "bash",
                            "-c",
                            f"""
export TF_CPP_MIN_LOG_LEVEL=2
mpiexec -N 1 pip install --user trl==0.17.0 vllm==0.8.5.post1 deepspeed==0.16.5 datasets loguru
mpiexec \
-x OMP_NUM_THREADS=1 \
-x MASTER_ADDR=$PFKUBE_PYTORCH_DIST_MASTER_ADDR \
-x MASTER_PORT=$PFKUBE_PYTORCH_DIST_MASTER_PORT \
-x PYTHONPATH=$(pwd) \
{accelerate_env_var} \
--bind-to none \
python /mnt/nfs-mnj-hot-99/tmp/hokuyama/jigsaw-agile-community-rules/train_{args.training_type}.py \
--model-name {args.model_name} \
--use-ompi \
--output-dir {Path(f"/mnt/nfs-mnj-hot-99/tmp/hokuyama/jigsaw-agile-community-rules/outputs/{args.workflow_name + "-" + args.model_name.replace(".","-").split("/")[-1]}-{start_time}")} \
--max-steps {args.max_steps}
""",
                        ],
                        working_dir="{{minai.git-snapshot}}",
                    )
                ],
                priority_class_name="low",
            ),
        ),
        slots=args.n_gpus,
        workers=args.n_nodes,
        retry_limit=0,
    ),
)
minai.basic.setup(mpi_job, cpu=32, gpu=args.n_gpus, gpu_name="a100", memory=(512, "Gi"))

minai.git_snapshot.setup(
    mpi_job, docker_repository=minai.git_snapshot.docker_repositories.minai_tmp()
)

main_template_name = minai.mpi_job.build(
    mpi_job=mpi_job,
    templates=templates,
    inject_pfkube_pytorch_dist=True,
)

workflow = minai.basic.create_workflow(
    name="jigzaw-" + args.training_type + "-" + args.workflow_name,
    spec=models.WorkflowSpec(
        templates=templates,
        entrypoint=main_template_name,
    ),
    activity_code="3000",
)

manifest = json.dumps(kubernetes_models.asdict(workflow))
subprocess.run("kubectl create -f -", input=manifest, shell=True, text=True, check=True)
