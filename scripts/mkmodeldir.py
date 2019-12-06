import argparse
import os
import yaml


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("target_dir")
    cli.add_argument("--comm-type")
    cli.add_argument("--force")
    cli.add_argument("--gru", action="store_true")
    opts = cli.parse_args()

    data = yaml.safe_load(open("scripts/model.yml"))
    if opts.comm_type is not None:
        data["model"]["use_communication"] = True
        data["model"]["comm_type"] = opts.comm_type
        data["model"]["use_gru"] = opts.gru
    else:
        data["model"]["use_communication"] = False
    os.makedirs(opts.target_dir, exist_ok=opts.force)
    data_str = yaml.dump(data)
    print(data_str)
    with open(os.path.join(opts.target_dir, "config.yml"), "w") as f:
        f.write(data_str)


if __name__ == "__main__":
    main()
