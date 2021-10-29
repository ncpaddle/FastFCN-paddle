import numpy as np
import torch
import paddle


def transfer():
    input_fp = "/Users/zhaoxing04/PaddleP/FastFCN/fastfcn.pth"
    output_fp = "./fastfcn_paddle.pdparams"
    torch_dict = torch.load(input_fp)
    paddle_dict = {}
    fc_names = [
        "classifier.1.weight", "classifier.4.weight", "classifier.6.weight"
    ]
    for key in torch_dict:
        print(key)
    #     weight = torch_dict[key].cpu().detach().numpy()
    #     flag = [i in key for i in fc_names]
    #     if any(flag):
    #         print("weight {} need to be trans".format(key))
    #         weight = weight.transpose()
    #     paddle_dict[key] = weight
    # paddle.save(paddle_dict, output_fp)


if __name__ == '__main__':
    transfer()