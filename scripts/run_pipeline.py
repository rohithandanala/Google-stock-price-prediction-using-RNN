import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.trainer.train_and_predict import train_and_predict



if __name__ == "__main__":
    train_and_predict()
