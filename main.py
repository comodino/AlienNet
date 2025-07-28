import argparse

def run_training():
    from training.train import run_training
    run_training()

def run_evaluation():
    from training.evaluate import run_evaluation
    run_evaluation()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlienNet CLI")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval'],
                        help="Modalità da eseguire: 'train' o 'eval'")
    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    elif args.mode == "eval":
        run_evaluation()
