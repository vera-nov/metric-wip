"""Load scaling/fusion/calibration weights and merge A/B/C JSON lists into D."""
import argparse
from file_scores import fuse_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", nargs="?", default="russian_scores.json", help="Russian GREEN scores")
    parser.add_argument("b", nargs="?", default="text_scores.json", help="English text scores")
    parser.add_argument("c", nargs="?", default="image_scores.json", help="Image/text scores")
    parser.add_argument("--weights", default="quality_weights.json")
    parser.add_argument("--output", default="final_scores.json")
    args = parser.parse_args()
    result = fuse_files(args.a, args.b, args.c, args.weights, args.output)
    print(f"Saved {len(result)} scores to {args.output}")


if __name__ == "__main__":
    main()
