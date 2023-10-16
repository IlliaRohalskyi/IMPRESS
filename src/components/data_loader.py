import os
import subprocess


def load_dvc():
    username = os.environ.get("DVC_USERNAME")
    token = os.environ.get("DVC_TOKEN")

    subprocess.run(["dvc", "remote", "modify", "origin", "--local", "auth", "basic"])
    subprocess.run(["dvc", "remote", "modify", "origin", "--local", "user", username])
    subprocess.run(["dvc", "remote", "modify", "origin", "--local", "password", token])

    subprocess.run(["dvc", "pull", "-r", "origin"])


if __name__ == "__main__":
    load_dvc()
