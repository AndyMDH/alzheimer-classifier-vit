import argparse
from src.models.train import train_model

def main():
    # Argument parsing for model selection
    parser = argparse.ArgumentParser(description='Train 3D ViT models')
    parser.add_argument('--model', type=str, default='vit_b16', help='Choose a model: vit_b16 or vit_m8')
    args = parser.parse_args()

    # Step 1: Train the model
    print(f"Starting training for model: {args.model}")
    train_model(model_name=args.model)  # Pass the chosen model name to the train function

if __name__ == "__main__":
    main()
