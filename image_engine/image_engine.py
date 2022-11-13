#!/usr/bin/python3

from libsql_utils.model.trade import formStrategySet, formInvestValue
import pandas as pd
from pandas import DataFrame as df
from pandas import DataFrame
import matplotlib.pyplot as plt
import io
import base64


def single_curve_plot(x, y, param: dict):
    """
    parameters requirements:\n
    1. length of x equals y;\n
    2. param must contains keys W, H, resolution in integer, title, xlabel, ylabel in string and image_type equals jpg or png.\n
    3. if save image into url param["output"] = url and url = param["url]\n
    4. if param["output]=bytes then ignore the url parameter.
    """
    # 设定图像plot属性
    plt.figure(figsize=(param["W"], param["H"]), dpi=param["resolution"])
    plt.plot(x, y, color="black", linestyle="solid")
    plt.title(param["title"])
    plt.xlabel(param["xlabel"])
    plt.ylabel(param["ylabel"])
    # 修饰图像数据
    if param["output"] == "url":
        image_data = plt.savefig(param["url"], format=param["image_type"])
    else:
        buffer = io.BytesIO()
        plt.savefig(buffer, format=param['image_type'])
        data = buffer.getvalue()
        data = base64.b64encode(data)
        image_data = bytes.decode(data)
    return image_data

if __name__ == '__main__':
    x = [0, 1, 2, 3]
    y = [1, 1.5, 2.8, 4.6]
    param = {"W": 12, "H": 4, "resolution": 72, "title": "Test", "xlabel": "xlabel", "ylabel": "ylabel", "output": "url", "image_type": "png", "url": "/home/fred/Documents/dev/image_engine/image_engine/test.png"}  
    param = {"W": 12, "H": 4, "resolution": 72, "title": "Test", "xlabel": "xlabel", "ylabel": "ylabel", "output": "bytes", "image_type": "png", "url": "/home/fred/Documents/dev/image_engine/image_engine/test.png"}
    single_curve_plot(x, y, param)