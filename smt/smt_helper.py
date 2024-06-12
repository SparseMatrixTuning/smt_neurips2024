import torch
import numpy as np
from collections import defaultdict
import heapq
import re
from helpers.deepspeed_helpers import print_rank_0
import deepspeed

global_rank = torch.distributed.get_rank()







def select_submatrix_based_on_grads(grads, n=660, selection_strategy = "no_restriction", calculate_strategy = "mean_abs", model = "yahma/llama-13b-hf"):
    """
    grad: grad information for each MLP Linear weight matrix
    n: number of sub-matrix to choose
    """
    # Step 1: Calculate absolute value of mean for all grad tensors in every 256x256 block
    if (model == "yahma/llama-13b-hf") or (model == "NousResearch/Llama-2-13b-hf"):
        Block_dimension = 256
        large_d = 54
        small_d = 20
    # elif model == "yahma/llama-7b-hf":
    elif (model == "NousResearch/Llama-2-7b-hf") or (model == "yahma/llama-7b-hf") or (model == "meta-llama/Llama-2-7b-chat-hf"):
        Block_dimension = 256
        large_d = 43
        small_d = 16
    elif (model == "NousResearch/Meta-Llama-3-8B") or (model == "meta-llama/Meta-Llama-3-8B"):
        Block_dimension = 256
        large_d = 56
        small_d = 16

    block_means = {}
    for key, grad in grads.items():
        # Reshape the grad tensor into 256x256 blocks
        if key[0] == 'gate_proj' or key[0] == 'up_proj':
            # print(key[0], grad.size())
            print_rank_0(f"gate_proj and up_proj dimension check:{key[0]}, {grad.size()}", global_rank)

            reshaped_grad = grad.reshape(large_d, Block_dimension, small_d, Block_dimension)

        elif key[0] == 'down_proj':
            # print(key[0], grad.size())
            print_rank_0(f"down_proj dimension check:{key[0]}, {grad.size()}", global_rank)

            reshaped_grad = grad.reshape(small_d, Block_dimension, large_d, Block_dimension)


        elif key[0] == 'q_proj' or key[0] == 'k_proj' or key[0] == 'v_proj':
            print_rank_0(f"qkv dimension check:{key[0]}, {grad.size()}", global_rank)
            if (model == "meta-llama/Meta-Llama-3-8B") and (key[0] == 'k_proj' or key[0] == 'v_proj'):
                small_d_ = 4
                reshaped_grad = grad.reshape(small_d_, Block_dimension, small_d, Block_dimension)
            else:
                reshaped_grad = grad.reshape(small_d, Block_dimension, small_d, Block_dimension)

    # print("tensor shape:", reshaped_grad.shape)
        if calculate_strategy == 'mean_abs':
            block_means[key] = mean_abs(reshaped_grad)
        elif calculate_strategy == 'abs_mean':
            block_means[key] = abs_mean_(reshaped_grad)
        elif calculate_strategy == 'L1':
            block_means[key] = L1_norm(reshaped_grad)
        elif calculate_strategy == 'L2':
            block_means[key] = L2_norm(reshaped_grad)



    # for each linear layer, select certain number of sub-matrix, normal distributed selection
    if selection_strategy == "norm_dist":
        # Step 2: Rank all the blocks in all grad tensors using the abs.mean() value
        ranked_blocks = defaultdict(list)

        for key, block_mean in block_means.items():
            indices = torch.argsort(block_mean.view(-1), descending=True)
            # print("===================================================")
            # print("indices", indices)
            top_indices = indices[:n]
            for idx in top_indices:
                # may need to consider int memory cost in the future
                row = idx // block_mean.shape[1]
                col = idx % block_mean.shape[1]
                ranked_blocks[key].append((row.item(), col.item()))
        del indices
        del top_indices
        del key
        del block_mean
        # Step 3: Return the selected blocks and their corresponding information
        return ranked_blocks

    else:
        # Step 2: Use a min-heap to maintain top n blocks efficiently
        top_blocks = []
        for key, block_mean in block_means.items():
            for i in range(block_mean.shape[0]):
                for j in range(block_mean.shape[1]):
                    abs_mean = block_mean[i, j].item()
                    if len(top_blocks) < n:
                        heapq.heappush(top_blocks, (abs_mean, (key, i, j)))
                    else:
                        heapq.heappushpop(top_blocks, (abs_mean, (key, i, j)))

        # print("===================================================")
        # print("top_blocks", top_blocks)

        # Step 3: Return the selected top n blocks and their corresponding information
        top_blocks.sort(reverse=True)  # Sort the top_blocks in descending order
        ranked_blocks = defaultdict(list)

        # selected_blocks = [(info, row, col, mean) for mean, (info, row, col) in top_blocks]


        # print("===================================================")
        # print("selected_blocks", selected_blocks)
        # for (info, row, col, mean) in selected_blocks:
        for mean, (info, row, col) in top_blocks:
            ranked_blocks[info].append((row, col))

        del top_blocks
        del mean
        del info
        del key
        del block_mean
        return ranked_blocks


def mean_abs(grad_tensor):
    print_rank_0(f"use mean()abs() as calculation strategy", global_rank)
    return grad_tensor.mean(dim=(1, 3)).abs()

def abs_mean_(grad_tensor):
    print_rank_0(f"use abs()mean() as calculation strategy", global_rank)
    return grad_tensor.abs().mean(dim=(1, 3))

def L1_norm(grad_tensor):
    print_rank_0(f"use L1 norm as calculation strategy", global_rank)

    return grad_tensor.abs().sum(dim=(1, 3))

def L2_norm(grad_tensor):
    print_rank_0(f"use L2 norm as calculation strategy", global_rank)
    return torch.sqrt(torch.sum(grad_tensor.abs() ** 2, dim=(1, 3)))

def replacement_test(ref_param, model_param, x_index, y_index):
    matrix_equal = torch.equal(ref_param, model_param)
    if matrix_equal:

        print("The ref_param and param have the same values.")
    else:
        print("The ref_param and param have different values.")
    print("====================================================")

    matrix_equal1 = torch.equal(ref_param[x_index: x_index + 256, y_index: y_index + 256],
                                model_param[x_index: x_index + 256, y_index: y_index + 256])
    if matrix_equal1:
        print("The selected region have the same values.")
    else:
        print("The selected region have different values.")





if __name__ == '__main__':


    # Example 1 usage:
    grads = {
        ('gate_proj', 1): torch.zeros(11008, 4096),
        ('up_proj', 1): torch.zeros(11008, 4096),
        ('down_proj', 2): torch.ones(4096, 11008)
    }

    grads[('gate_proj', 1)][0:256, 0:256] = torch.ones(256, 256)*10
    grads[('gate_proj', 1)][256:512, 0:256] = torch.ones(256, 256)*10
    grads[('up_proj', 1)][0:2560, 0:256] = torch.ones(2560, 256)*10
    selected_blocks = select_submatrix_based_on_grads(grads, n=20, selection_strategy = "no_restriction")
    print(selected_blocks)
    print(len(selected_blocks))
