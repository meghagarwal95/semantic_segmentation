from train import Trainer
import argparse
import models

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)

    parser.add_argument('--learning_rate', type=float, default=0.01)

    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--n_epochs', type=int, default=10)

    parser.add_argument('--root_folder',type=str, default="./../../datasets/mit_scene_parsing/ADEChallengeData2016/")

    return parser

if __name__=="__main__":

    parser = get_parser().parse_args()
    trainer = Trainer(parser.root_folder, batch_size=parser.batch_size, learning_rate=parser.learning_rate, n_epochs=parser.n_epochs, momentum=parser.momentum)
    trainer.run()
