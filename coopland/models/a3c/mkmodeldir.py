import argparse
import shutil
import os


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("target_dir")
    opts = cli.parse_args()

    os.makedirs(opts.target_dir, exist_ok=False)
    shutil.copyfile(
        "coopland/models/a3c/model.yml", os.path.join(opts.target_dir, "config.yml")
    )


if __name__ == "__main__":
    main()
