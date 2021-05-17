from dpdi.helper.helper import Helper
from dpdi.helper.image_helper import ImageHelper


def get_helper(params, d, name):
    helper = ImageHelper(current_time=d, params=params, name=name)
    return helper
