import json
import subprocess
import argparse

import kubernetes_models
import minai
from minai import models

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
args = parser.parse_args()

# Make ArgoのContainer Template
main_template = models.Template(
    name="main",
    container=models.Container(
        name="",
        # set all-in-one image
        image="asia-northeast1-docker.pkg.dev/pfn-artifactregistry/all-in-one/stable:latest",
        command=[
                "bash",
                "-c",
                f"""
export TF_CPP_MIN_LOG_LEVEL=2
export PATH=$PATH:/home/hokuyama/.local/bin
pip install cairosvg autoawq==0.2.8
python /mnt/nfs-mnj-hot-99/tmp/hokuyama/make-data-count/auto_awq.py
"""
        ],
    ),
)

minai.basic.setup(main_template, cpu=64, gpu=1, gpu_name="a100", memory=(512, "Gi"))

workflow = minai.basic.create_workflow(
    name="auto-awq",
    spec=models.WorkflowSpec(
        templates=[main_template],
        entrypoint=main_template.name,
    ),
    activity_code="3000",
)

manifest = json.dumps(kubernetes_models.asdict(workflow))
subprocess.run("kubectl create -f -", input=manifest, shell=True, text=True, check=True)
