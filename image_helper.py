
from collections import defaultdict

import torch
import torch.utils.data

from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np

from models.simple import SimpleNet


logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0



class ImageHelper(Helper):


    def poison(self):
        return

    def load_data(self):

        return

    def create_model(self):
        return