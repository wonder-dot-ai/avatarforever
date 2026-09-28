"""Experimental two-stage AR inference using the released Avatar-Forever checkpoint.

Uses the existing sampler: half-resolution AR generation, 2x latent upsampling,
then full-resolution AR refinement. This does not stream chunks between stages.
All ordinary inference.py options are supported, including a reference image.
"""

from pathlib import Path

from inference import build_arg_parser, run_inference
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES


def main() -> None:
    parser = build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(num_frames=257, output_dir=Path(__file__).resolve().parent / "videos" / "two-stage")
    parser.add_argument("--spatial-upsampler-path", type=Path, required=True, help="LTX-2.3 spatial x2 weights.")
    parser.add_argument("--stage2-sigmas", type=float, nargs="+", default=STAGE_2_DISTILLED_SIGMA_VALUES)
    args = parser.parse_args()
    if not args.spatial_upsampler_path.is_file():
        parser.error("--spatial-upsampler-path must point to an existing checkpoint")
    if args.height <= 0 or args.width <= 0 or args.height % 64 or args.width % 64:
        parser.error("Two-stage output height and width must be positive multiples of 64")
    run_inference(
        args,
        stage_mode="two-stage",
        spatial_upsampler_path=args.spatial_upsampler_path,
        stage2_sigmas=args.stage2_sigmas,
    )


if __name__ == "__main__":
    main()
