import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
torch.cuda.current_device()
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import _utils
from utils.load import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular').cuda()
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu').cuda()

    def forward(self, x):
        # x (batch_size, window_history, c_in) => (batch_size, d_model, window_history) => (:, window_history, d_model)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2) 
        return x #(B, L, D)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float().to(device) # (max_len, d_model)
        pe.require_grad = False

        position = torch.arange(0, max_len).float().to(device).unsqueeze(1) # (max_len, 1)
        div_term = (torch.arange(0, d_model, 2).float().to(device) * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :] # (1, window_history, d_model) = (1, L, D)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in = c_in, d_model = d_model)
        self.position_embedding = PositionalEmbedding(d_model = d_model)

        self.dropout = nn.Dropout(p = dropout).cuda()

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(x) # (batch_size, window_history, d_model) = (B, L, D)

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout).cuda()

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys).to(device)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1)).to(device)
        V = torch.einsum("bhls,bshd->blhd", A, values).to(device)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention_old(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout).cuda()

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E).to(device)
        index_sample = torch.randint(L_K, (L_Q, sample_k)).to(device) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :].to(device)
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze().to(device)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :].to(device) # factor*ln
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)).to(device) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2).to(device)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone().to(device)
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex.to(device)

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1).to(device) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in).to(device)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1) # (B, H, L, D)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.contiguous(), attn

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout).cuda()

    def _prob_QK(self, Q, K, sample_k, n_top, flag, weight=0.9): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape # (125, 8, 24, E) initially
        _, _, L_Q, _ = Q.shape
        #print(K.shape, Q.shape)
        if flag == 'prob':

            # calculate the sampled Q_K
            K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E).to(device) # 125 8 24 24 E
            index_sample = torch.randint(L_K, (L_Q, sample_k)).to(device) # real U = U_part(factor*ln(L_k))*L_q
            K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :].to(device)
            Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze().to(device)
        
            # top k kurtosis
            Kur = (((Q_K_sample - Q_K_sample.mean(dim=-1, keepdim=True)) / Q_K_sample.std(dim=-1, keepdim=True))**4).mean(-1)

            # find the Top_k query with sparisty measurement
            M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)        
            M_Kur = weight * M + (1 - weight) * Kur        

            # use the reduced Q to calculate Q_K
            Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :].to(device) # factor*ln
            Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)).to(device) # factor*ln(L_q)*L_k
        else:
            Q_K = torch.matmul(Q, K.transpose(-2, -1)).to(device) # factor*ln(L_q)*L_k
            M_top = Q[:, :, :, 0].topk(Q.shape[-2], sorted=False)[1]

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2).to(device)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone().to(device)
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex.to(device)

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1).to(device) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in).to(device)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1) # (B, H, L, D)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q)

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        flag = 'prob' if u<L_Q else 'skip'
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u, flag=flag) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.contiguous(), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads).cuda()
        self.key_projection = nn.Linear(d_model, d_keys * n_heads).cuda()
        self.value_projection = nn.Linear(d_model, d_values * n_heads).cuda()
        self.out_projection = nn.Linear(d_values * n_heads, d_model).cuda()
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn # (B, L, D)

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular').cuda()
        self.norm = nn.BatchNorm1d(c_in).cuda()
        self.activation = nn.ELU().cuda()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1).cuda()

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1)) # (B, D, L) -> (B, D, L+4)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x) # (B, D, (L+5)/2)
        x = x.transpose(1,2)
        return x # if lossless, (B, L, D)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1).cuda()
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1).cuda()
        self.norm1 = nn.LayerNorm(d_model).to(device)
        self.norm2 = nn.LayerNorm(d_model).to(device)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x) # (B, L, D)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn # (B, L, D)

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None): 
        # attn_layers is a set of EncoderLayer, conv_layers is a set of ConvLayer
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1).cuda()
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1).cuda()
        self.norm1 = nn.LayerNorm(d_model).to(device)
        self.norm2 = nn.LayerNorm(d_model).to(device)
        self.norm3 = nn.LayerNorm(d_model).to(device)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
        
class Aqua_lstm(nn.Module):
    def __init__(self, PSR, configs):
        super(Aqua_lstm, self).__init__()
        self.num_layers = 2
        d = configs.d_model
        c_in = configs.enc_in * PSR[1]
        c_out = configs.c_out
        self.lstm = nn.LSTM(c_in, d, num_layers=self.num_layers, dropout = 0.1, bidirectional=True, batch_first = True).cuda()
        self.proj = nn.Linear(d*2, c_out, bias=True).cuda()
    def forward(self, x_enc, x_dec, lstm, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x = x_enc
        self.lstm.flatten_parameters()
        self.lstm.activation = nn.ReLU()
        out, _ = self.lstm(x)
        out = self.proj(out[:, -1, :]).unsqueeze(1)
        return out, out

class Aquaformer(nn.Module):
    def __init__(self, PSR, enc_in, dec_in, c_out, seq_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', activation='gelu', 
                output_attention = False, distil=True, 
                device=torch.device('cuda:0')):
        super(Aquaformer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        enc_in = enc_in * PSR[1]
        dec_in = enc_in

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model).to(device)
        )
        # Decoder_mask
        self.decoder_mask = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model).to(device)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model).to(device)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True).cuda()
        
    def forward(self, x_enc, x_dec, lstm, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        first, _ = lstm(x_enc, x_enc, x_enc)
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        if x_enc.shape == x_dec.shape:
            dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        else:
            dec_out = self.decoder_mask(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        prime = dec_out[:, -self.pred_len, :].unsqueeze(1)-first
        dec_out = dec_out[:,-self.pred_len:,:]-prime
        out = torch.cat((first, dec_out[:, 1:, :]), dim=1)
        
        if self.output_attention:
            return out, attns
        else:
            return out # [B, L, D]

class TriangularCausalMask():
    def __init__(self, B, L, device=device):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device=device):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask