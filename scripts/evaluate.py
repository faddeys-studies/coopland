import argparse
import os
import numpy as np


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("model_dir_or_results_txt")
    cli.add_argument("--stuck-thr", "-T", type=int)
    cli.add_argument("--maze-size", "-S", type=int)
    opts = cli.parse_args()

    pth = opts.model_dir_or_results_txt
    if os.path.isdir(pth):
        if opts.maze_size is None:
            from coopland.models.a3c import config_lib
            from coopland.utils import load_from_yml

            cfg = load_from_yml(config_lib.ModelConfig, os.path.join(pth, "config.yml"))
            opts.maze_size = cfg.maze_size
        pth = os.path.join(pth, "test-results.txt")

    results = np.loadtxt(pth, int)
    if results.ndim == 1:
        results = np.expand_dims(results, 1)

    res_max = results.max(axis=1)
    print("N:", results.shape[0])
    print("Max-Mean:", res_max.mean())
    print("Min-Mean:", results.min(axis=1).mean())
    print("Mean:", results.mean())

    stuck_thr = opts.stuck_thr if opts.stuck_thr is not None else results.max()
    print("Stuck rate:", np.sum(res_max >= stuck_thr) / results.shape[0])

    res_exited = results[res_max < stuck_thr]
    print("Exited Max-Mean:", res_exited.max(axis=1).mean())
    print("Exited Min-Mean:", res_exited.min(axis=1).mean())
    print("Exited Mean:", res_exited.mean())

    if opts.maze_size is not None:
        maze_area = opts.maze_size * opts.maze_size
        res_max = np.minimum(res_max, stuck_thr).astype("uint")
        prob_dens = (
            np.histogram(res_max, bins=stuck_thr, range=(1, stuck_thr))[0]
            / res_max.size
        )
        prob = np.cumsum(prob_dens)
        x = np.exp(-np.square(np.arange(stuck_thr) / maze_area) / 2)
        sorter = np.argsort(x)
        x = x[sorter]
        prob = prob[sorter]
        x[0] = 0
        etc_auc = np.sum((prob[1:] + prob[:-1]) * (x[1:] - x[:-1]) / 2)
        print("ETC-AUC:", etc_auc)


if __name__ == "__main__":
    main()
