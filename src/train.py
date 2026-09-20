import os

from src.model import build_model


def main():
    # Placeholder train script. Hook dataset loading and training here.
    vocab_size = 20000
    model = build_model(vocab_size=vocab_size)
    os.makedirs('models', exist_ok=True)
    model.save('models/toxic_model.keras')


if __name__ == '__main__':
    main()
