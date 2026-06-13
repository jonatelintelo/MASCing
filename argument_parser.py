import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--root", default="", type=str)
    parser.add_argument("--print_logging", action="store_true")

    arguments = parser.parse_args()

    print(f"\nConfiguration: {vars(arguments)}")

    return arguments
