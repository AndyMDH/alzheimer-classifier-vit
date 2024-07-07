import subprocess

models = ["vit_3d.py", "resnet_3d.py", "densenet_3d.py"]

for model in models:
    print(f"Running {model}")
    subprocess.run(["python", model])