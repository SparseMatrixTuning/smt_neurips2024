
from deepspeed.compression.helper import recursive_getattr, recursive_setattr
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from deepspeed.ops.adam.multi_tensor_apply import MultiTensorApply
import torch
from torch import nn
import torch.nn.functional as F
import re
import unittest
from pytorch_memlab import MemReporter
from collections import defaultdict
import deepspeed
import torch
import numpy as np
import heapq
import re
from helpers.deepspeed_helpers import print_rank_0
import deepspeed

deepspeed.init_distributed()
global_rank = torch.distributed.get_rank()
Block_dimension = 256





def convert_linear_layer_to_matrix_sparsity(model, selected_submatrix, selected_submatrix_attention, part_module_name=['.layers']):
    pattern = re.compile(r'model\.layers\.(\d+)\.')

    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(part in name for part in part_module_name):
            # print(f"convert {name} to LoRA")
            replace_name.append(name)
    for name in replace_name:
        if "mlp" in name:
            module = recursive_getattr(model, name)
            if module.weight.requires_grad:
                print_rank_0(f"Module Test: {name}", global_rank)
                module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
                match = pattern.search(name)
                layer_number = int(match.group(1)) if match else None

                # index_list: list of index which require_grad, need to pass into Linear
                index_list = selected_submatrix[(module_name, layer_number)]

                tmp = LinearLayer_MatrixSparsity(
                    module.weight,
                    bias = None,
                    index_list = index_list).to(module.weight.device).to(module.weight.dtype)
                recursive_setattr(model, name, tmp)
        if "self_attn" in name:
            module = recursive_getattr(model, name)
            if module.weight.requires_grad:
                print_rank_0(f"Module Test: {name}", global_rank)
                module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else None
                match = pattern.search(name)
                layer_number = int(match.group(1)) if match else None

                # index_list: list of index which require_grad, need to pass into Linear
                index_list = selected_submatrix_attention[(module_name, layer_number)]

                tmp = LinearLayer_MatrixSparsity(
                    module.weight,
                    bias = None,
                    index_list = index_list).to(module.weight.device).to(module.weight.dtype)
                recursive_setattr(model, name, tmp)


    return model



########################## SparseLinear -> forward, backward #####################

class LinearLayer_MatrixSparsity(torch.nn.Module):
    # an simple implementation of matrix sparsity
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 bias=None,
                 index_list = []):
        super(LinearLayer_MatrixSparsity, self).__init__()
        self.weight = weight
        self.weight.requires_grad = False
        self.bias = bias
        self.index_list = index_list


        self.selected_weight = torch.empty(len(index_list) * Block_dimension, Block_dimension,dtype=self.weight.data.dtype,
                                  device=self.weight.data.device)

        for i in range(len(index_list)):
            index = index_list[i]
            self.selected_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :] = self.weight.data[index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, index[1] * Block_dimension: index[1] * Block_dimension + Block_dimension]
        self.selected_weight.requires_grad = True
        self.selected_weight = nn.Parameter(self.selected_weight)


        self.fn = linearZ.apply

    def forward(self, x):
        for i in range(len(self.index_list)):
            index = self.index_list[i]
            # self.selected_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :] = self.weight.data[index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, index[1] * Block_dimension: index[1] * Block_dimension + Block_dimension]
            self.weight.data[index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, index[1] * Block_dimension: index[1] * Block_dimension + Block_dimension] = self.selected_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :]

        x = self.fn(x,  self.selected_weight, self.index_list, self.weight)
        return x

class linearZ(torch.autograd.Function):
    # only support batch size D=3 now, for batch size = 1, need to add mm. operation.
    @staticmethod
    def forward(ctx, input, selected_weight, matrix_index_list, weight):
        input_list = []
        for index in matrix_index_list:
            input_list.append(input[:, :, index[1]*Block_dimension: index[1]*Block_dimension+Block_dimension])
        # save for backward may only support tensor, use others to save!
        ctx.list1 = input_list
        ctx.list2 = matrix_index_list

        ctx.save_for_backward(weight)


        # output = input.mm(weight.t())
        # print("input size:",input.size())
        # print("weight size:",weight.data.size())
        output = torch.matmul(input, weight.t())


        # memory free
        del weight
        del input_list
        del matrix_index_list


        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight,  = ctx.saved_tensors
        input_list = ctx.list1
        matrix_index_list = ctx.list2

        # Pytorch use C++ engine to check whether gradient has matched dimenstion or not
        grad_weight = torch.empty(len(input_list) * Block_dimension, Block_dimension,dtype=grad_output.dtype,
                                  device=grad_output.device)
        for i in range(len(input_list)):
            index = matrix_index_list[i]

            # print("index:", index)
            # print("grad_output_dim:", grad_output.size())
            # tmp = grad_output.permute(0, 2, 1)[:, index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, :]
            # print("tmp size", tmp.size())
            # print("input list[i]", input_list[i].size())
            # tmp1 = torch.matmul(tmp, input_list[i])
            # grad_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :] = torch.sum(tmp1, dim=0)

            grad_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :] = torch.sum(torch.matmul(grad_output.permute(0, 2, 1)[:, index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, :], input_list[i]), dim=0)

        grad_input = torch.matmul(grad_output, weight)

        # memory free
        del weight
        del input_list
        del matrix_index_list

        return grad_input, grad_weight, None, None


######################### utils ###############################


# Source Code from DeepSpeed Examples official website
# Please refer to https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-Chat/dschat/utils/utils.py#L210

def get_optimizer_sparse_grouped_parameters(
    model,
    weight_decay,
    smt_lr,
    lora_lr=5e-4,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):


    print_rank_0(f"================ PRINT PARAM NAME [0]=======================", global_rank)

    for name, param in model.named_parameters():
        if (not any(nd in name.lower() for nd in no_decay_name_list)
                and param.requires_grad and not any(nd in name.lower() for nd in lora_name_list)):
            print_rank_0(f"name0:{name}", global_rank)

    print_rank_0(f"================ PRINT PARAM NAME [1]=======================", global_rank)
    for n, p in model.named_parameters():
        if (not any(nd in n.lower() for nd in no_decay_name_list)
                and p.requires_grad and any(nd in n.lower() for nd in lora_name_list)):
            print_rank_0(f"name1:{n}", global_rank)



    print_rank_0(f"================ PRINT PARAM NAME [2]=======================", global_rank)
    for n, p in model.named_parameters():
        if (any(nd in n.lower() for nd in no_decay_name_list) and p.requires_grad):
            print_rank_0(f"name2:{n}", global_rank)



    optimizer_grouped_parameters = [
        {
            "params": #tmp
            [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower() for nd in lora_name_list))
            ]
            ,
            "weight_decay":
            weight_decay,
            "lr":
            smt_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower() for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)

    print_rank_0(f"group parameters: {non_empty_groups}", global_rank)

    return non_empty_groups #, sorted_selected_submatrix




def freeze_unselected_matrix_layer(model, select_parameters, select_attention_parameters, mixture = False):
    # selected_parameters: (module_name, layer_number, head_number)
    # model = convert_selected_sau_to_linear_layer(model, select_parameters, exclude)
    pattern = re.compile(r'model\.layers\.(\d+)\.')
    for name, param in model.named_parameters():
        if mixture:
            if "mlp" in name:
                module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
                match = pattern.search(name)
                layer_number = int(match.group(1)) if match else None
                if (module_name, layer_number) in select_parameters.keys():
                    param.requires_grad = True
                    print_rank_0(f"Layer set to grad = True:{name}", global_rank)

                    # print("selected grad True layer")
                    # print(module_name, layer_number)
                else:
                    param.requires_grad = False
                    print_rank_0(f"Layer set to grad = Flase:{name}", global_rank)

            elif "self_attn" in name:
                module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else None
                match = pattern.search(name)
                layer_number = int(match.group(1)) if match else None
                if (module_name, layer_number) in select_parameters.keys():
                    param.requires_grad = True
                    print_rank_0(f"Layer set to grad = True:{name}", global_rank)


                else:
                    param.requires_grad = False
                    print_rank_0(f"Layer set to grad = Flase:{name}", global_rank)

            else:
                param.requires_grad = False
                print_rank_0(f"Layer set to grad = False:{name}", global_rank)

        else:
            if "mlp" in name:
                module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
                match = pattern.search(name)
                layer_number = int(match.group(1)) if match else None
                if (module_name, layer_number) in select_parameters.keys():
                    param.requires_grad = True
                    print_rank_0(f"Layer set to grad = True:{name}", global_rank)

                    # print("selected grad True layer")
                    # print(module_name, layer_number)
                else:
                    param.requires_grad = False
                    print_rank_0(f"Layer set to grad = Flase:{name}", global_rank)

            elif "self_attn" in name:
                module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else None
                match = pattern.search(name)
                layer_number = int(match.group(1)) if match else None
                if (module_name, layer_number) in select_attention_parameters.keys():
                    param.requires_grad = True
                    print_rank_0(f"Layer set to grad = True:{name}", global_rank)


                else:
                    param.requires_grad = False
                    print_rank_0(f"Layer set to grad = Flase:{name}", global_rank)

            else:
                param.requires_grad = False
                print_rank_0(f"Layer set to grad = False:{name}", global_rank)

    return model







########################## Test Code ###################################

# example code for
class MyLinearZ(nn.Module):
    def __init__(self):
        super(MyLinearZ, self).__init__()
        self.fn = linearZ.apply
        self.weight = nn.Parameter(torch.randn(1, 1) * 5.66)
    def forward(self, x):
        x = self.fn(x, self.weight)
        return x


def memoryTest():
    DEVICE = 'cpu'
    weight = torch.randn(4096, 11008).to(DEVICE)
    weight2 = torch.zeros(4096, 11008)
    weight3 = torch.randn(256, 256*6)
    weight2[0:256, 0:256*6] = weight3
    weight4 = weight2.to_sparse().to(DEVICE)

    del weight2
    del weight3
    torch.cuda.empty_cache()

    reporter = MemReporter()
    reporter.report()




def fwbwTest():
    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0")  # Uncomment this to run on GPU

    # Create Tensors to hold input and outputs.
    # By default, requires_grad=False, which indicates that we do not need to
    # compute gradients with respect to these Tensors during the backward pass.

    # Define input and weight tensors
    input = torch.randn(16, 2560, 2560)
    weight = torch.randn(2560, 2560)
    selected_weight = torch.randn(256*2, 256)
    weight.requires_grad = True

    # Define index list for matrix sparsity
    index_list = [(0, 0), (0, 1)]  # Example index list

    # Create random Tensors for weights. For this example, we need
    # 4 weights: y = a + b * P3(c + d * x), these weights need to be initialized
    # not too far from the correct result to ensure convergence.
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.

    learning_rate = 5e-6
    for t in range(2000):
        # To apply our Function, we use Function.apply method. We alias this as 'P3'.
        P3 = linearZ.apply

        # Forward pass: compute predicted y using operations; we compute
        # P3 using our custom autograd operation.
        y_pred = P3(input, selected_weight, index_list, weight)

        # Compute and print loss
        loss = (y_pred - input).sum()
        print(t, loss.item())

        # Use autograd to compute the backward pass.
        loss.backward()


if __name__ == '__main__':
    # memoryTest()
    fwbwTest()

