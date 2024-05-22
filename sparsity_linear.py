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
from utils.utils import print_rank_0
import deepspeed


deepspeed.init_distributed()
global_rank = torch.distributed.get_rank()
Block_dimension = 256


def convert_linear_layer_to_matrix_sparsity(model, selected_submatrix, selected_submatrix_attention, part_module_name=['.layers']):
    pattern = re.compile(r'model\.layers\.(\d+)\.')

    # for name, module in model.named_modules():
    #     print(name, module)
    #     if module.weight.requires_grad == False:
    #         continue
    #     else:
    #         if "mlp" in name:
    #             module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
    #             match = pattern.search(name)
    #             layer_number = int(match.group(1)) if match else None
    #
    #             # index_list: list of index which require_grad, need to pass into Linear
    #             # can either try lora/dora print to debug...
    #             index_list = selected_submatrix[(module_name, layer_number)]
    #             module = recursive_getattr(model, name)

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


############################ Optimizers ########################################


class FusedMatrixSparseAdam(FusedAdam):

    def __init__(self,
                 params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 adam_w_mode=True,
                 weight_decay=0.,
                 amsgrad=False,
                 set_grad_none=True):

        super(FusedAdam, self).__init__(params,
                 lr=lr,
                 bias_correction=bias_correction,
                 betas=betas,
                 eps=eps,
                 adam_w_mode=adam_w_mode,
                 weight_decay=weight_decay,
                 amsgrad=amsgrad,
                 set_grad_none=set_grad_none)
        self.multi_tensor_applier = MultiTensorApply(2048 * 32)
        print("================param groups==================")
        print(self.param_groups)


    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError(
                'FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments.'
            )
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if len(group['params']) == 0:
                continue
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' not in group:
                group['step'] = 0

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_bf, p_bf, m_bf, v_bf = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        'FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # DeepSpeed ZeRO 3 processes each subgroup a time, so we need to keep tracking step count for each tensor separately.
                    # While this is not an issue for ZeRO 1 & 2, since they apply a single optimization step to the whole param group at the same time.
                    # In order to keep backward compatibility for the existing checkpoints, we use group['state'] to initialize state['step'] if it exists.
                    state['step'] = group.get('step', 0)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                    v_16.append(state['exp_avg_sq'])
                elif p.dtype == torch.bfloat16:
                    g_bf.append(p.grad)
                    p_bf.append(p)
                    m_bf.append(state['exp_avg'])
                    v_bf.append(state['exp_avg_sq'])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])
                else:
                    raise RuntimeError('FusedAdam only support fp16, bf16 and fp32.')

            if len(g_16) > 0:
                state['step'] += 1
                self.multi_tensor_applier(self.multi_tensor_adam, self._dummy_overflow_buf, [g_16, p_16, m_16, v_16],
                                     group['lr'], beta1, beta2, group['eps'], state['step'], self.adam_w_mode,
                                     bias_correction, group['weight_decay'])

            if len(g_bf) > 0:
                state['step'] += 1
                self.multi_tensor_applier(self.multi_tensor_adam, self._dummy_overflow_buf, [g_bf, p_bf, m_bf, v_bf],
                                     group['lr'], beta1, beta2, group['eps'], state['step'], self.adam_w_mode,
                                     bias_correction, group['weight_decay'])

            if len(g_32) > 0:
                state['step'] += 1
                self.multi_tensor_applier(self.multi_tensor_adam, self._dummy_overflow_buf, [g_32, p_32, m_32, v_32],
                                     group['lr'], beta1, beta2, group['eps'], state['step'], self.adam_w_mode,
                                     bias_correction, group['weight_decay'])

        return loss





class DeepSpeedCPUMatrixSparisityAdam(DeepSpeedCPUAdam):
    optimizer_id = 0

    def __init__(self,
                 model_params,
                 sorted_selected_submatrix = None,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 adamw_mode=True,
                 fp32_optimizer_states=True):

        super(DeepSpeedCPUMatrixSparisityAdam, self).__init__(model_params,
                 lr=lr,
                 bias_correction=bias_correction,
                 betas=betas,
                 eps=eps,
                 weight_decay=weight_decay,
                 amsgrad=amsgrad,
                 adamw_mode=adamw_mode,
                 fp32_optimizer_states=fp32_optimizer_states)
        self.sorted_selected_submatrix = sorted_selected_submatrix
        print("================param groups==================")
        print(self.param_groups)



    @torch.no_grad()
    def step(self, closure=None, fp16_param_groups=None):
        """Update the model parameters.

        .. note::
            This method will be called internally by ZeRO-Offload. DeepSpeed
            users should still use ``engine.step()`` as shown in the
            `Getting Started
            <https://www.deepspeed.ai/getting-started/#training>`_ guide.

        Args:
            closure (callable, optional): closure to compute the loss.
                Defaults to ``None``.
            fp16_param_groups: FP16 GPU parameters to update. Performing the
                copy here reduces communication time. Defaults to ``None``.

        Returns:
            loss: if ``closure`` is provided. Otherwise ``None``.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # intended device for step
        device = torch.device('cpu')

        # converting the fp16 params to a group of parameter
        if type(fp16_param_groups) is list:
            if type(fp16_param_groups[0]) is not list:
                fp16_param_groups = [fp16_param_groups]
        elif fp16_param_groups is not None:
            fp16_param_groups = [[fp16_param_groups]]

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):

                if p.grad is None:
                    continue

                assert p.device == device, f"CPUAdam param is on {p.device} and must be 'cpu', make " \
                        "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    #print(f'group {group_id} param {param_id} = {p.numel()}')
                    state['step'] = 0

                    #use full precision by default unless self.fp32_optimizer_states is off
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype

                    # gradient momentums
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    #memory_format=torch.preserve_format)
                    # gradient variances
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    #memory_format=torch.preserve_format)

                state['step'] += 1
                beta1, beta2 = group['betas']

                if fp16_param_groups is not None:
                    self.ds_opt_adam.adam_update_copy(self.opt_id, state['step'], group['lr'], beta1, beta2,
                                                      group['eps'], group['weight_decay'], group['bias_correction'],
                                                      p.data, p.grad.data, state['exp_avg'], state['exp_avg_sq'],
                                                      fp16_param_groups[group_id][param_id].data)
                else:
                    self.ds_opt_adam.adam_update(self.opt_id, state['step'], group['lr'], beta1, beta2, group['eps'],
                                                 group['weight_decay'], group['bias_correction'], p.data, p.grad.data,
                                                 state['exp_avg'], state['exp_avg_sq'])
        return loss


######################### utils ###############################

def get_optimizer_sparse_grouped_parameters(
    model,
    weight_decay,
    selected_submatrix,
    lora_lr=5e-4,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
    lora_name_list=["lora_right_weight", "lora_left_weight", "sau_weights", 'magnitude'],
):
    # pattern = re.compile(r'model\.layers\.(\d+)\.')
    # tmp = list()
    # sorted_selected_submatrix = list()
    # for name, param in model.named_parameters():
    #     if (not any(nd in name.lower() for nd in no_decay_name_list)
    #             and param.requires_grad and not any(nd in name.lower() for nd in lora_name_list)):
    #
    #         # Need to debug
    #         if "mlp" in name:
    #             module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
    #             match = pattern.search(name)
    #             layer_number = int(match.group(1)) if match else None
    #             print(name)
    #             print(param)
    #             # index_list: list of index which require_grad, need to pass into Linear
    #             # can either try lora/dora print to debug...
    #             index_list = selected_submatrix[(module_name, layer_number)]
    #             print("=====================TEST======================")
    #             print((module_name, layer_number))
    #             print(index_list)
    #             # weight parameters
    #             param_weight = torch.empty(Block_dimension, Block_dimension * len(index_list),dtype=param.dtype,
    #                                       device=param.device, requires_grad=True)
    #
    #             #weight gradients
    #             grad_weight = torch.empty(Block_dimension, Block_dimension * len(index_list),dtype=param.dtype,
    #                                       device=param.device)
    #             for i in range(len(index_list)):
    #                 index = index_list[i]
    #                 print(param.data)
    #                 print(param.data.size())
    #                 print(index[0], index[1])
    #                 test = param.data[index[0]*Block_dimension:(index[0]*Block_dimension+Block_dimension), index[1]*Block_dimension:(index[1]*Block_dimension+Block_dimension)]
    #                 print(test)
    #                 print(param.data.size())
    #                 print("print grad:", param.data.grad)
    #                 print("print require_grad:", param.data.requires_grad)
    #                 # grad_weight[0:256, i * 256: i * 256 + 256] = param.data.grad[index[0] * 256: index[0] * 256 + 256, index[1] * 256: index[1] * 256 + 256]
    #
    #                 with torch.no_grad():
    #                     param_weight[0:Block_dimension, i * Block_dimension : i*Block_dimension + Block_dimension] = param.data[index[0]*Block_dimension:(index[0]*Block_dimension+Block_dimension), index[1]*Block_dimension:(index[1]*Block_dimension+Block_dimension)]
    #                     # grad_weight[0:256, i * 256: i * 256 + 256] = param.data.grad[index[0] * 256: index[0] * 256 + 256, index[1] * 256: index[1] * 256 + 256]
    #
    #             # param_weight.grad = grad_weight
    #             param_weight_ = nn.Parameter(param_weight)
    #
    #             sorted_selected_submatrix.append(index_list)
    #             tmp.append(param_weight_)

    print_rank_0(f"================ PRINT PARAM NAME [0]=======================", global_rank)

    for name, param in model.named_parameters():
        if (not any(nd in name.lower() for nd in no_decay_name_list)
                and param.requires_grad and not any(nd in name.lower() for nd in lora_name_list)):
            print_rank_0(f"name0:{name}", global_rank)

    print_rank_0(f"================ PRINT PARAM NAME [1]=======================", global_rank)
    for n, p in model.named_parameters():
        if (not any(nd in n.lower() for nd in no_decay_name_list)
                and p.requires_grad and any(nd in n.lower()
                                            for nd in lora_name_list)):
            print_rank_0(f"name1:{name}", global_rank)



    print_rank_0(f"================ PRINT PARAM NAME [2]=======================", global_rank)
    for n, p in model.named_parameters():
        if (any(nd in n.lower()
                for nd in no_decay_name_list) and p.requires_grad):
            print_rank_0(f"name2:{name}", global_rank)



    optimizer_grouped_parameters = [
        {
            "params": #tmp
            [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower()
                                                    for nd in lora_name_list))
            ]
            ,
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower()
                                                for nd in lora_name_list))
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
    return non_empty_groups #, sorted_selected_submatrix




def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == deepspeed.runtime.zero.
        partition_parameters.ZeroParamStatus.NOT_AVAILABLE
    ]




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


def sparseOperationTimeTest():

    pass


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

    # .device=device, dtype=dtype, requires_grad=True)

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

        # Update weights using gradient descent
        # with torch.no_grad():
        #     a -= learning_rate * a.grad
        #     b -= learning_rate * b.grad
        #     c -= learning_rate * c.grad
        #     d -= learning_rate * d.grad
        #
        #     # Manually zero the gradients after updating weights
        #     a.grad = None
        #     b.grad = None
        #     c.grad = None
        #     d.grad = None

    # print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')




if __name__ == '__main__':
    # memoryTest()
    fwbwTest()


