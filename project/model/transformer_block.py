import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
       
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads  # dimensión por cabeza
        
        # Proyecciones lineales para Q, K, V
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        
        # Proyección final después de concatenar todas las cabezas
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x): # Que pasa con los datos al llamar al módulo 
        """
        x: (B, T, d_model)
        Devuelve: (B, T, d_model)
        """
        B, T, _ = x.size()
        
        # 1. Proyecciones lineales
        Q = self.Wq(x)  # (B, T, d_model)
        K = self.Wk(x)  # (B, T, d_model)
        V = self.Wv(x)  # (B, T, d_model)
        
        # 2. Reorganizar en múltiples cabezas
        # (B, T, d_model) -> (B, num_heads, T, d_k)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Calcular scores de atención
        # (B, num_heads, T, d_k) x (B, num_heads, d_k, T) -> (B, num_heads, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 4. Máscara causal (triangular superior a -inf)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        # 5. Normalizar con softmax
        with torch.amp.autocast(enabled=False):
            scores = scores.float()
            attn = torch.softmax(scores, dim=-1)  # (B, num_heads, T, T)
        
        # 6. Aplicar atención a V
        out = torch.matmul(attn, V)  # (B, num_heads, T, d_k)
        
        # 7. Recombinar heads
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, d_model)
        
        # 8. Proyección final
        out = self.Wo(out)  # (B, T, d_model)
        return out
    


# Add & norm


class NormLayer(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))  
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, X):
        # X: (batch, seq_len, hidden_dim)
        with torch.amp.autocast(enabled=False): # Desactivo mixed precision manualmente pues es recomendable en la LayerNorm y como la he implementado yo, quiza no lo detecta automaticamente (solo reconoce capas nativas de pytorch) [35]
            X = X.float()
            mean = X.mean(dim=-1, keepdim=True)       # media por posición
            var = X.var(dim=-1, keepdim=True, unbiased=False)  # varianza
            X_hat = (X - mean) / torch.sqrt(var + self.eps)    # normaliza
        return self.gamma * X_hat + self.beta


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout) # Para regularizar
        self.ln = NormLayer(norm_shape) # nn.LayerNorm(norm_shape)

    def forward(self, X, Y): # Y es la salida de la subcapa previa y X la entrada a la subcapa
        return self.ln(self.dropout(Y) + X) # Aplica add y luego layernorm

# FFNN

class FeedForward(nn.Module):
    def __init__(self,d_model, d_ff, dropout):
        super().__init__()
        # Se usan 3 capas densas, con dropout y activación GELU
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_ff)
        self.linear3 = nn.Linear(d_ff, d_model) # Ver si estas capas expanden y contraen
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
       x = self.linear1(x)
       x = self.gelu(x)
       x = self.dropout(x)
       x = self.linear2(x)
       x = self.gelu(x)
       x = self.dropout(x)
       x = self.linear3(x)
       return x

# Bloque 1 transformer decoder-only

class TransformerDecoderOnlyBlock(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, d_model)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)
        self.apply(self._init_weights) # Recorre las capas aplicando la función

    def _init_weights(self, m): # Inicialización de pesos: Xavier para MHA y FFNN y normal para embeddings
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias) # Para evitar desplazamiento inicial arbitrario
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self,x):
        attention = self.mha(x)
        x = self.addnorm1(x, attention)
        ffn_out = self.ffn(x)
        x = self.addnorm2(x, ffn_out)
        return x
