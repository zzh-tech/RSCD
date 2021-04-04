from para import Parameter
from train import Trainer

if __name__ == '__main__':
    para = Parameter().args
    trainer = Trainer(para)
    trainer.run()

