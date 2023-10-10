import argparse, os, time, json
from src import text_bert, exp_configs, utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_mode",
        nargs="+",
        default="text",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mm-reddit",
        help="dataset",
    )
    parser.add_argument(
        "-s", 
        "--split",
        type=int,
        default=0,
        help='kfolds-split to use'
    )
    args, unknown = parser.parse_known_args()

    # use curr timestamp to create a unique folder to save a run
    output_dir = os.path.join("./checkpoints/", utils.hash_str(str(time.time())) + '-split-' + str(args.split))
    os.makedirs(output_dir)
    os.makedirs('logs/' + output_dir.split('/')[-1])
    if args.exp_mode == ["text"]:
        exp_dict = exp_configs.get_base_config(dname=args.dataset, modalities="text")
        exp_dict["output_dir"] = output_dir
        utils.wandb_init(exp_dict)
        with open(os.path.join(output_dir, "exp_dict.json"), "w") as f:
            json.dump(exp_dict, f)
        text_bert.main(exp_dict, args.split)
    else:
        raise ValueError(f"exp_mode={args.exp_mode} not supported")
