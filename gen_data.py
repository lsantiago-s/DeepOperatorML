import os
import sys
import argparse
import logging
from src.problems import ProblemRegistry
from src.problems.datagen_plotting import generate_problem_dataset_plots
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

def gen_data() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, type=str, help="Problem name")
    parser.add_argument(
        "--config",
        required=False,
        default=None,
        type=str,
        help="Optional explicit datagen config path. Defaults to configs/problems/<problem>/datagen.yaml",
    )
    args = parser.parse_args()

    config_path = args.config or os.path.join("./configs/problems/", args.problem, 'datagen.yaml')

    try:
        generator = ProblemRegistry.get_generator(name=args.problem, config=config_path)
        generator.generate()
        data_filename = getattr(generator, "config", {}).get("data_filename")
        if data_filename:
            try:
                generate_problem_dataset_plots(
                    problem_name=args.problem,
                    data_path=data_filename,
                )
            except Exception as exc:
                logger.warning(
                    "Dataset generation succeeded, but sanity plot generation failed for %s: %s",
                    args.problem,
                    exc,
                )
        else:
            logger.info("Skipping dataset sanity plots for %s: generator config has no data_filename.", args.problem)
    except KeyError:
        logging.error(f"Unknown problem: {args.problem}")
        raise

if __name__ == "__main__":
    gen_data()
