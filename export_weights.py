"""Export compact inference files from an existing fitted method, without retraining."""
import argparse
from file_scores import export_artifact
from pipeline import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", default="quality_method_config.json")
    parser.add_argument("--weights", default="quality_weights.json")
    parser.add_argument("--components", default="quality_components.json")
    args = parser.parse_args()
    export_artifact(read_json(args.artifact), args.weights, args.components)
    print(f"Saved {args.weights} and {args.components}")


if __name__ == "__main__":
    main()
