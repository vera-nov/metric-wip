"""Inference from the fitted artifact, using only the selected base components."""
import argparse
from pipeline import infer, read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', '--artifact', dest='config', required=True)
    p.add_argument('--path-to-data-description-for-evaluation', '--input', dest='input', required=True)
    p.add_argument('--output', default='quality_scores.json')
    p.add_argument('--work-dir', required=True)
    p.add_argument('--execution-config', help='Optional JSON to relocate environments, GPU and image_root')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    infer(read_json(args.config), args.input, args.work_dir, args.output,
          read_json(args.execution_config) if args.execution_config else None,
          args.resume, args.dry_run)


if __name__ == '__main__':
    main()
