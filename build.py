from parts import *



class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,
                 source_token_embed:TokenEmbedding,target_token_embed:TokenEmbedding,
                 source_position_embed:PositionEmbedding,target_position_embed:PositionEmbedding,
                 projector:ProjectionLayer):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.source_token_embed=source_token_embed
        self.target_token_embed=target_token_embed
        self.source_position_embed=source_position_embed
        self.target_position_embed=target_position_embed
        self.projector=projector

    def encode(self,source,source_mask):
        source=self.source_token_embed(source)
        source=self.source_position_embed(source)
        result=self.encoder(source,source_mask)
        return result

    def decode(self,now_output,encoder_output,target_mask,source_mask):
        now_output=self.target_token_embed(now_output)
        now_output=self.target_position_embed(now_output)
        result=self.decoder(now_output,encoder_output,target_mask,source_mask)
        return result

    def project(self,x):
        result=self.projector(x)
        return result


def build_transformer(d_model:int,d_hidden:int,
                      num_heads:int,drop_prob:float,
                      num_encode_layers:int,num_decode_layers:int,
                      source_vocab_size:int,target_vocab_size:int,
                      source_seq_len:int,target_seq_len:int,
                      tie_target_embedding:bool=False
                      )->Transformer:
    #build the outer tools
    source_token_embed=TokenEmbedding(source_vocab_size,d_model)
    target_token_embed=TokenEmbedding(target_vocab_size,d_model)
    source_position_embed=PositionEmbedding(source_seq_len,d_model,drop_prob)
    target_position_embed=PositionEmbedding(target_seq_len,d_model,drop_prob)

    #build the encoder
    encoder_layers=[]
    for _ in range(num_encode_layers):
        encode_self_attention_block=MultiHeadAttention(d_model,num_heads,drop_prob)
        encode_feed_forward_block=FeedForwardBlock(d_model,d_hidden,drop_prob)
        encoder_layer=EncoderLayer(encode_self_attention_block,encode_feed_forward_block,d_model,drop_prob)
        encoder_layers.append(encoder_layer)
    encoder_layers=nn.ModuleList(encoder_layers)
    encoder=Encoder(encoder_layers)

    #build the decoder
    decoder_layers=[]
    for _ in range(num_decode_layers):
        decode_self_attention_block=MultiHeadAttention(d_model,num_heads,drop_prob)
        cross_attention_block=MultiHeadAttention(d_model,num_heads,drop_prob)
        decode_feed_forward_block=FeedForwardBlock(d_model,d_hidden,drop_prob)
        decoder_layer=DecoderLayer(decode_self_attention_block,cross_attention_block,decode_feed_forward_block,d_model,drop_prob)
        decoder_layers.append(decoder_layer)
    decoder_layers=nn.ModuleList(decoder_layers)
    decoder=Decoder(decoder_layers)

    #build the projector
    projector=ProjectionLayer(d_model,target_vocab_size)

    # Weight tying improves sample efficiency for low-resource translation and
    # keeps the public projection API unchanged.
    if tie_target_embedding:
        projector.w_proj.weight=target_token_embed.embedding.weight

    #create the whole Transformer!
    transformer=Transformer(encoder,decoder,source_token_embed,target_token_embed,source_position_embed,target_position_embed,projector)

    #initialize the parameters
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return transformer
