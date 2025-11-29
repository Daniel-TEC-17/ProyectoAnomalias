from trainer import Trainer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train a machine learning model.')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    trainer = Trainer(config_path=args.config)
    trainer.run()

if __name__ == '__main__':
    main()