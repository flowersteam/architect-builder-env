from env_comem.com_channel.channel import Channel
import readchar
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_size', type=int, default=5)
    parser.add_argument('--type', type=str, default='identity', choices=['identity', 'one-hot'])
    return parser.parse_args()


def _run_channel(args):
    chnl = Channel(args.dict_size, type=args.type, seed=1223)
    arrow_keys = {str(i): i for i in range(chnl.dict_size)}

    while True:
        key = readchar.readkey()
        if key not in arrow_keys.keys():
            break

        chnl.send(arrow_keys[key])
        print(chnl.read())
        chnl.render(True)


if __name__ == "__main__":
    args = get_args()
    _run_channel(args)
