"""Evaluate one selected component on a JSON list, retaining every input field."""
import argparse
from file_scores import COMPONENTS, DEFAULT_OUTPUTS, score_component


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=COMPONENTS)
    parser.add_argument("input", help="JSON list with reference_text, candidate_text, reference_image_path and optional unique string report_id")
    parser.add_argument("--config", default="quality_components.json")
    parser.add_argument("--output", help="Defaults to A.json / B.json / C.json")
    parser.add_argument("--work-dir", default="work/evaluation")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = score_component(args.component, args.input, args.config, output=args.output,
                             work_dir=args.work_dir, resume=not args.no_resume, dry_run=args.dry_run)
    if result is not None:
        print(f"Saved {len(result)} scores to {args.output or DEFAULT_OUTPUTS[args.component]}")


if __name__ == "__main__":
    main()
