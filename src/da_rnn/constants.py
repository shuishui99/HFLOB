import torch
import sys
sys.path.append('/home/yuan/jam/pycharm/src/')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
