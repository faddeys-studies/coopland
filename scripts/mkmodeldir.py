#!/usr/bin/env python
import argparse
import os
import yaml


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("target_dir")
    cli.add_argument("--comm-type")
    cli.add_argument("--force", action="store_true")
    cli.add_argument("--gru", action="store_true")
    cli.add_argument("--max-agents", type=int)
    cli.add_argument("--can-see", action="store_true")
    cli.add_argument("--only-hear", action="store_false", dest="can_see")
    opts = cli.parse_args()

    data = yaml.safe_load(open("scripts/model.yml"))
    if opts.comm_type is not None:
        if opts.comm_type != "none":
            data["model"].setdefault("comm", {})
            data["model"]["comm"]["type"] = opts.comm_type
            data["model"]["comm"]["use_gru"] = opts.gru
        else:
            data["model"]["comm"] = None
    if opts.max_agents is not None:
        data["model"]["max_agents"] = opts.max_agents
    if opts.can_see is not None:
        data["model"]["comm"]["can_see_others"] = opts.can_see
    os.makedirs(opts.target_dir, exist_ok=opts.force)
    data_str = yaml.dump(data)
    print(data_str)
    with open(os.path.join(opts.target_dir, "config.yml"), "w") as f:
        f.write(data_str)


if __name__ == "__main__":
    main()
