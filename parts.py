import torch
import torch.nn as nn
import math



#Token编码部分
class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size:int,d_model:int):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("vocab_size and d_model must be positive")
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        x=self.embedding(x) * math.sqrt(self.d_model)
        return x


#位置编码部分
class PositionEmbedding(nn.Module):
    def __init__(self,seq_len:int,d_model:int,drop_prob:float):
        super().__init__()
        if seq_len <= 0 or d_model <= 0:
            raise ValueError("seq_len and d_model must be positive")
        self.seq_len=seq_len
        self.d_model=d_model
        self.dropout = nn.Dropout(drop_prob)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        # Slice div_term so odd d_model values are supported as well.
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        pe = pe.unsqueeze(0) 
        self.register_buffer("pe", pe)
        
    def forward(self,x):
        if x.shape[1] > self.seq_len:
            raise ValueError(
                f"sequence length {x.shape[1]} exceeds configured maximum {self.seq_len}"
            )
        x=x+self.pe[:,:x.shape[1], :] # type: ignore
        return self.dropout(x)
        

#归一化部分
class LayerNormalization(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-6):
        super().__init__()
        self.eps=eps
        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zeros(d_model))

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        variance=x.var(dim=-1,keepdim=True,correction=0)
        std=torch.sqrt(variance+self.eps)

        x=(x-mean)/std
        x=self.gamma*x+self.beta
        return x


#残差连接部分
class ResidualConnection(nn.Module):
    def __init__(self,d_model:int,drop_prob:float):
        super().__init__()
        self.norm=LayerNormalization(d_model)
        self.dropout=nn.Dropout(drop_prob)

    def forward(self,x,sub_process):
        x=x+self.dropout(sub_process(x))
        x=self.norm(x)
        return x


#前馈部分
class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,d_hidden:int,drop_prob:float):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_hidden)
        self.dropout=nn.Dropout(drop_prob)
        self.linear2=nn.Linear(d_hidden,d_model)
    
    def forward(self,x):
        x=self.linear1(x)
        x=torch.relu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        return x
    
    

#多头注意力部分
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,drop_prob:float):
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model//num_heads

        self.w_q=nn.Linear(d_model,d_model,bias=False)
        self.w_k=nn.Linear(d_model,d_model,bias=False)
        self.w_v=nn.Linear(d_model,d_model,bias=False)
        self.w_o=nn.Linear(d_model,d_model,bias=False)
        
        self.dropout=nn.Dropout(drop_prob)
    
    def forward(self,q,k,v,mask):
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)

        #shape=[batch_size,seq_len,d_model]-->[batch_size,num_heads,seq_len,head_dim]
        query=query.view(query.shape[0],query.shape[1],self.num_heads,self.head_dim).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.num_heads,self.head_dim).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.num_heads,self.head_dim).transpose(1,2)

        x=MultiHeadAttention.attention(query,key,value,mask,self.dropout)

        #x-->shape=[batch_size,seq_len,d_model]
        x=x.transpose(1,2).contiguous()
        x=x.view(x.shape[0],x.shape[1],self.d_model)
        x=self.w_o(x)
        return x

    @staticmethod
    def attention(query,key,value,mask,dropout):
        score=query@key.transpose(-2,-1)
        head_dim=query.shape[-1]
        score=score/math.sqrt(head_dim)

        if mask is not None:
            score=score.masked_fill(mask.to(dtype=torch.bool),torch.finfo(score.dtype).min)

        score=score.softmax(dim=-1)
        score=dropout(score)

        result=score@value
        return result


#编码器层
class EncoderLayer(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttention,feed_forward_block:FeedForwardBlock,d_model:int,drop_prob:float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connection_list=nn.ModuleList([ResidualConnection(d_model,drop_prob) for _ in range(2)])

    def forward(self,x,source_mask):
        x=self.residual_connection_list[0](x,lambda a:self.self_attention_block(a,a,a,source_mask))
        x=self.residual_connection_list[1](x,self.feed_forward_block)
        return x


#完整编码器
class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers

    def forward(self,x,source_mask):
        for layer in self.layers:
            x=layer(x,source_mask)
        return x


#解码器层
class DecoderLayer(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttention,cross_attention_block:MultiHeadAttention,
                 feed_forward_block:FeedForwardBlock,
                 d_model:int,drop_prob:float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connection_list=nn.ModuleList([ResidualConnection(d_model,drop_prob) for _ in range(3)])

    def forward(self,now_output,encoder_output,target_mask,source_mask):
        self_attention_result=self.residual_connection_list[0](now_output,lambda a:self.self_attention_block(a,a,a,target_mask))

        cross_attention_result=self.residual_connection_list[1](self_attention_result,lambda a:self.cross_attention_block(a,encoder_output,encoder_output,source_mask))
        decoder_layer_output=self.residual_connection_list[2](cross_attention_result,self.feed_forward_block)
        return decoder_layer_output


#完整解码器
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers

    def forward(self,now_output,encoder_output,target_mask,source_mask):
        for layer in self.layers:
            now_output=layer(now_output,encoder_output,target_mask,source_mask)
        return now_output


#投影层
class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.w_proj=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        logits=self.w_proj(x)
        return logits
