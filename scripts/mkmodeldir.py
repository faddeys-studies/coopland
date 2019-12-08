#!/usr/bin/env python
import argparse
import os
import yaml


def _bool_t(s):
    s = s.lower().strip()
    if s in 'yes true t 1':
        return True
    if s in 'no false f 0':
        return False
    raise ValueError(s)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("target_dir")
    cli.add_argument("--comm-type", required=True)
    cli.add_argument("--force", action="store_true")
    cli.add_argument("--gru", action="store_true")
    cli.add_argument("--max-agents", type=int, required=True)
    cli.add_argument("--can-see", type=_bool_t, required=True)
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
