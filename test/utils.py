from datetime import datetime
import numpy as np


def get_date_time(formatted=False):
    if formatted:
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    return datetime.now().strftime("%d-%m-%Y_%HH%M")


def link_pos_to_state(link):
    link = np.array(link).flatten()
    return link
