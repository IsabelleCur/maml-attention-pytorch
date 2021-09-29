import math
from enum import Enum
import torch
import torch.nn as nn
from complexFunctions import complex_matmul
from complexLayers import ComplexLinear, ComplexDropout, NaiveComplexLayerNorm

class MixingMatrixInit(Enum):
    CONCATENATE = 1
    ALL_ONES = 2
    UNIFORM = 3


class CollaborativeAttention(nn.Module):
    def __init__(
        self,
        dim_input: int,#256
        dim_value_all: int,#user_dense_layer和dim_output必须相等 64
        dim_key_query_all: int,#都是int 64
        dim_output: int,#第2项 256
        num_attention_heads: int,#8
        output_attentions: bool,#flase
        attention_probs_dropout_prob: float,#0.1
        use_dense_layer: bool,#flase
        use_layer_norm: bool,#true
        mixing_initialization: MixingMatrixInit = MixingMatrixInit.UNIFORM,
    ):
        super().__init__()

        if dim_value_all % num_attention_heads != 0:
            raise ValueError(
                "Value dimension ({}) should be divisible by number of heads ({})".format(
                    dim_value_all, num_attention_heads
                )
            )

        if not use_dense_layer and dim_value_all != dim_output:
            raise ValueError(
                "Output dimension ({}) should be equal to value dimension ({}) if no dense layer is used".format(
                    dim_output, dim_value_all
                )
            )

        # save args
        self.dim_input = dim_input
        self.dim_value_all = dim_value_all
        self.dim_key_query_all = dim_key_query_all
        self.dim_output = dim_output
        self.num_attention_heads = num_attention_heads
        self.output_attentions = output_attentions
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.mixing_initialization = mixing_initialization
        self.use_dense_layer = use_dense_layer
        self.use_layer_norm = use_layer_norm

        self.dim_value_per_head = dim_value_all // num_attention_heads#整数除法
        self.attention_head_size = (
            dim_key_query_all / num_attention_heads
        )  # does not have to be integer

        # intialize parameters
        self.query = ComplexLinear(dim_input, dim_key_query_all, bias=False)#定义了一个query函数 输出维度是key和query的总维度dim_key_query_all, 输入维度是dim_input
        self.key = ComplexLinear(dim_input, dim_key_query_all, bias=False)
        self.content_bias = ComplexLinear(dim_input, num_attention_heads, bias=False)
        self.value = ComplexLinear(dim_input, dim_value_all)#

        self.mixing = self.init_mixing_matrix()#已改成复数

        self.dense = (
            ComplexLinear(dim_value_all, dim_output) if use_dense_layer else nn.Sequential()
        )#input: all,output: dim_output

        self.dropout = ComplexDropout(attention_probs_dropout_prob)

        if use_layer_norm:
            self.layer_norm = NaiveComplexLayerNorm(dim_value_all)

    def forward(
        self,
        hidden_states,#这是？？？
        attention_mask=None,#好像全都要是复数
        head_mask=None,#复
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        #print('---!!!---')
        #print(hidden_states.size())
        from_sequence = hidden_states#dim_input
        to_sequence = hidden_states#dim_input
        if encoder_hidden_states is not None:
            to_sequence = encoder_hidden_states
            attention_mask = encoder_attention_mask

        query_layer = self.query(from_sequence)#总维度，相当于展平 dim_all
        key_layer = self.key(to_sequence)
        #print(query_layer.size())
        #print(self.mixing.size())
        # point wise multiplication of the mixing coefficient per head with the shared query projection
        # (batch, from_seq, dim) x (head, dim) -> (batch, head, from_seq, dim)
        #(batch, 1, from_seq, dim)*(head, 1, dim)
        mixed_query_r = query_layer[..., None, :, :].real * self.mixing[..., :, None, :].real-query_layer[..., None, :, :].imag * self.mixing[..., :, None, :].imag
        mixed_query_i = query_layer[..., None, :, :].real * self.mixing[..., :, None, :].imag + query_layer[..., None, :,:].imag*self.mixing[..., :, None, :].real
        mixed_query=mixed_query_r.type(torch.complex64)+1j*mixed_query_i.type(torch.complex64)
        #mixed_query = query_layer[..., None, :, :]
        #print(mixed_query.size())
        #(batch, head, from_seq, dim)

        # broadcast the shared key for all the heads

        # (batch, 1, to_seq, dim)
        mixed_key = key_layer[..., None, :, :]#...和-1一样，该是多少是多少，None在左右之间增加一维
        #print(mixed_key.size())
        #mixed_key_t=mixed_key.transpose(-1, -2)
        # (batch, head, from_seq, to_seq)
        #a_r=mixed_query.real*mixed_key_t.real-mixed_query.imag*mixed_key_t.imag
        #a_i=mixed_query.real*mixed_key_t.imag+mixed_query.imag*mixed_key_t.real
        attention_scores = complex_matmul(mixed_query,mixed_key.transpose(-1, -2))
        #print("a1", attention_scores.size())

        # add the content bias term
        # (batch, to_seq, heads)#这个linear我怎么感觉是在最后加了一个dim
        content_bias = self.content_bias(to_sequence)#已改成复数
        # (batch, heads, 1, to_seq)
        broadcast_content_bias = content_bias.transpose(-1, -2).unsqueeze(-2)
        attention_scores += broadcast_content_bias#from_seq每一个加bias
        #print("a2", attention_scores.size())

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.跑不动了
        attention_probs = nn.Softmax(dim=-1)(attention_scores.real).type(torch.complex64)+1j*nn.Softmax(dim=-1)(attention_scores.imag).type(torch.complex64)#改成复数：实+虚 【0，1】
        #print("a3", attention_scores.size())
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)#已改成复数
        #print(attention_probs.size())

        # Mask heads if we want to
        if head_mask is not None:
            a__r=attention_probs.real*head_mask.real-attention_probs.imag*head_mask.imag
            a__i=attention_probs.real*head_mask.imag+attention_probs.imag*head_mask.real
            attention_probs = a__r.type(torch.complex64)+1j*a__i.type(torch.complex64)

        value_layer = self.value(to_sequence)
        value_layer = self.transpose_for_scores(value_layer)#更改形状
        #print("value_layer:", value_layer.size())
        #c_r=attention_probs.real*value_layer.real-attention_probs.imag*value_layer.imag
        #c_i=attention_probs.real*value_layer.imag+attention_probs.imag*value_layer.real
        context_layer = complex_matmul(attention_probs,value_layer)
        #print(context_layer.size())
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()#换回来？
        new_context_layer_shape = context_layer.size()[:-2] + (self.dim_value_all,)
        context_layer = context_layer.view(*new_context_layer_shape)
        #print(context_layer.size())
        context_layer = self.dense(context_layer)
        #print(context_layer.size())
        if self.use_layer_norm:
            context_layer = self.layer_norm(from_sequence + context_layer)

        if self.output_attentions:
            return (context_layer, attention_probs)#tuple!
        else:
            return (context_layer,)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def init_mixing_matrix(self, scale=0.2):
        #print(self.num_attention_heads)
        #print(self.dim_key_query_all)
        mixing = torch.zeros(self.num_attention_heads, self.dim_key_query_all)#2-dim
        #print(mixing.type())
        mixingi = torch.zeros(self.num_attention_heads, self.dim_key_query_all)
        #print(mixing.type())
        if self.mixing_initialization is MixingMatrixInit.CONCATENATE:
            # last head will be smaller if not equally divisible
            dim_head = int(math.ceil(self.dim_key_query_all / self.num_attention_heads))
            for i in range(self.num_attention_heads):
                mixing[i, i * dim_head : (i + 1) * dim_head] = 1.0
                mixingi[i, i * dim_head: (i + 1) * dim_head] = 1.0
            #print(mixing.type())
        elif self.mixing_initialization is MixingMatrixInit.ALL_ONES:
            mixing.one_()
            mixingi.one_()
        elif self.mixing_initialization is MixingMatrixInit.UNIFORM:
            mixing.normal_(std=scale)
            mixingi.normal_(std=scale)
        else:
            raise ValueError(
                "Unknown mixing matrix initialization: {}".format(
                    self.mixing_initialization
                )
            )

        return nn.Parameter(mixing.type(torch.complex64)+1j*mixingi.type(torch.complex64))
