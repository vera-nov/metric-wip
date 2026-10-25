"""Build the paper's metric using persistent, environment-isolated feature stages."""
import argparse
from pipeline import fit, load_plan, prepare, resolved_plan, run_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True, help='JSON with models, columns and conda environments')
    p.add_argument('--work-dir', required=True, help='Persistent directory for intermediate CSVs')
    p.add_argument('--output-json', default='quality_method_config.json')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--features-only', action='store_true')
    group.add_argument('--fit-only', action='store_true')
    args = p.parse_args()
    plan = resolved_plan(load_plan(args.plan))
    prepare(plan, args.work_dir, require_target=True)
    if not args.fit_only:
        run_features(plan, args.work_dir, resume=args.resume, dry_run=args.dry_run)
    if not args.features_only and not args.dry_run:
        fit(plan, args.work_dir, args.output_json)


if __name__ == '__main__':
    main()
