import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy

def prune_and_restructure(model, pruning_rate=0.5):
    """
    Aplica pruning estrutural e remonta o modelo SimpleModelWithFlashAttention
    com menos parâmetros.
    """
    model_copy = copy.deepcopy(model)
    
    # Coletar parâmetros originais
    in_features = model.conv1[0].in_channels
    num_classes = model.fc.out_features
    dim_original = model.feature_dim
    n_heads = model.n_heads
    
    # Lista para armazenar as novas camadas
    new_layers = []
    masks = []
    
    # 1. PRUNNING DA conv1
    conv1_module = model_copy.conv1[0]
    prune.ln_structured(conv1_module, name='weight', amount=pruning_rate, n=2, dim=0)
    masks.append(list(model_copy.buffers())[0])
    prune.remove(conv1_module, 'weight')
    
    # Filtrar canais de saída da conv1
    indices_conv1_out = conv1_module.weight.abs().sum(dim=(1, 2, 3)) != 0
    n_conv1_out = indices_conv1_out.sum().item()
    
    # Criar nova conv1
    new_conv1_weight = conv1_module.weight[indices_conv1_out, :, :, :]
    new_conv1_bias = conv1_module.bias[indices_conv1_out]
    
    new_conv1 = nn.Conv2d(
        in_channels=in_features,
        out_channels=n_conv1_out,
        kernel_size=conv1_module.kernel_size,
        stride=conv1_module.stride,
        padding=conv1_module.padding,
        bias=True
    )
    new_conv1.weight = nn.Parameter(new_conv1_weight)
    new_conv1.bias = nn.Parameter(new_conv1_bias)
    
    new_layers.append(new_conv1)
    new_layers.append(nn.ReLU(inplace=True))
    new_layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
    
    # 2. PRUNNING DA conv2
    conv2_module = model_copy.conv2[0]
    prune.ln_structured(conv2_module, name='weight', amount=pruning_rate, n=2, dim=0)
    masks.append(list(model_copy.buffers())[0])
    prune.remove(conv2_module, 'weight')
    
    # Filtrar canais de saída da conv2
    indices_conv2_out = conv2_module.weight.abs().sum(dim=(1, 2, 3)) != 0
    n_conv2_out = indices_conv2_out.sum().item()
    
    # IMPORTANTE: Ajustar n_conv2_out para ser múltiplo de n_heads
    # Isso é necessário para a atenção funcionar
    n_conv2_out = (n_conv2_out // n_heads) * n_heads
    if n_conv2_out == 0:
        n_conv2_out = n_heads  # Mínimo
    
    # Filtrar pesos da conv2 considerando a conv1 pruned
    conv2_weight_pruned = conv2_module.weight[indices_conv2_out, :, :, :]
    conv2_weight_pruned = conv2_weight_pruned[:, indices_conv1_out, :, :]
    conv2_weight_pruned = conv2_weight_pruned[:n_conv2_out, :, :, :]  # Ajustar para múltiplo de n_heads
    
    conv2_bias_pruned = conv2_module.bias[indices_conv2_out]
    conv2_bias_pruned = conv2_bias_pruned[:n_conv2_out]
    
    # Criar nova conv2
    new_conv2 = nn.Conv2d(
        in_channels=n_conv1_out,
        out_channels=n_conv2_out,
        kernel_size=conv2_module.kernel_size,
        stride=conv2_module.stride,
        padding=conv2_module.padding,
        bias=True
    )
    new_conv2.weight = nn.Parameter(conv2_weight_pruned)
    new_conv2.bias = nn.Parameter(conv2_bias_pruned)
    
    new_layers.append(new_conv2)
    new_layers.append(nn.ReLU(inplace=True))
    new_layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
    
    # 3. CAMADAS DE ATENÇÃO - ajustar para nova dimensão
    # Nova dimensão será n_conv2_out (já ajustada para múltiplo de n_heads)
    new_dim = n_conv2_out
    new_head_dim = new_dim // n_heads
    
    # to_qkv: Linear(new_dim, new_dim * 3)
    to_qkv_original = model_copy.to_qkv
    # Redimensionar pesos para nova dimensão
    # Como a dimensão mudou, precisamos redimensionar os pesos apropriadamente
    to_qkv_weight = to_qkv_original.weight.data
    to_qkv_bias = to_qkv_original.bias.data
    
    # Se a dimensão original for maior que a nova, truncamos
    # Se for menor, adicionamos zeros (ou inicialização aleatória)
    if dim_original >= new_dim:
        # Truncar pesos e bias
        new_to_qkv_weight = to_qkv_weight[:new_dim*3, :new_dim]
        new_to_qkv_bias = to_qkv_bias[:new_dim*3]
    else:
        # Expandir com inicialização aleatória
        new_to_qkv_weight = torch.zeros((new_dim*3, new_dim))
        new_to_qkv_bias = torch.zeros(new_dim*3)
        new_to_qkv_weight[:dim_original*3, :dim_original] = to_qkv_weight
        new_to_qkv_bias[:dim_original*3] = to_qkv_bias
    
    new_to_qkv = nn.Linear(new_dim, new_dim * 3)
    new_to_qkv.weight = nn.Parameter(new_to_qkv_weight)
    new_to_qkv.bias = nn.Parameter(new_to_qkv_bias)
    new_to_qkv.is_attention = True
    
    # attention_out: Linear(new_dim, new_dim)
    attention_out_original = model_copy.attention_out
    attention_out_weight = attention_out_original.weight.data
    attention_out_bias = attention_out_original.bias.data
    
    if dim_original >= new_dim:
        new_attention_out_weight = attention_out_weight[:new_dim, :new_dim]
        new_attention_out_bias = attention_out_bias[:new_dim]
    else:
        new_attention_out_weight = torch.zeros((new_dim, new_dim))
        new_attention_out_bias = torch.zeros(new_dim)
        new_attention_out_weight[:dim_original, :dim_original] = attention_out_weight
        new_attention_out_bias[:dim_original] = attention_out_bias
    
    new_attention_out = nn.Linear(new_dim, new_dim)
    new_attention_out.weight = nn.Parameter(new_attention_out_weight)
    new_attention_out.bias = nn.Parameter(new_attention_out_bias)
    new_attention_out.is_attention = True
    
    # 4. CAMADAS FC
    # fc1: Linear(new_dim, 512)
    fc1_linear = model_copy.fc1[0]
    prune.ln_structured(fc1_linear, name='weight', amount=pruning_rate, n=2, dim=1)
    masks.append(list(model_copy.buffers())[0])
    prune.remove(fc1_linear, 'weight')
    
    # Filtrar dimensões da fc1
    # A entrada deve ser new_dim, não a dim_original
    fc1_weight_pruned = fc1_linear.weight.data
    fc1_bias_pruned = fc1_linear.bias.data
    
    # Primeiro, ajustar para a nova dimensão de entrada (new_dim)
    if dim_original >= new_dim:
        fc1_weight_pruned = fc1_weight_pruned[:, :new_dim]
    else:
        # Expandir com zeros
        temp_weight = torch.zeros((fc1_weight_pruned.size(0), new_dim))
        temp_weight[:, :dim_original] = fc1_weight_pruned
        fc1_weight_pruned = temp_weight
    
    # Agora aplicar pruning na dimensão de saída (se desejado)
    indices_fc1_out = fc1_weight_pruned.abs().sum(dim=1) != 0
    n_fc1_out = indices_fc1_out.sum().item()
    
    fc1_weight_pruned = fc1_weight_pruned[indices_fc1_out, :]
    fc1_bias_pruned = fc1_bias_pruned[indices_fc1_out]
    
    new_fc1_linear = nn.Linear(new_dim, n_fc1_out)
    new_fc1_linear.weight = nn.Parameter(fc1_weight_pruned)
    new_fc1_linear.bias = nn.Parameter(fc1_bias_pruned)
    
    # 5. CAMADA FINAL (fc) - não aplicar pruning pesado na última camada
    fc_original = model_copy.fc
    # Aplicar pruning mais leve na última camada
    prune.ln_structured(fc_original, name='weight', amount=pruning_rate/2, n=2, dim=1)
    masks.append(list(model_copy.buffers())[0])
    prune.remove(fc_original, 'weight')
    
    fc_weight_pruned = fc_original.weight.data
    fc_bias_pruned = fc_original.bias.data
    
    # Ajustar para entrada de tamanho n_fc1_out
    if fc_weight_pruned.size(1) >= n_fc1_out:
        fc_weight_pruned = fc_weight_pruned[:, :n_fc1_out]
    else:
        temp_weight = torch.zeros((fc_weight_pruned.size(0), n_fc1_out))
        temp_weight[:, :fc_weight_pruned.size(1)] = fc_weight_pruned
        fc_weight_pruned = temp_weight
    
    new_fc = nn.Linear(n_fc1_out, num_classes)
    new_fc.weight = nn.Parameter(fc_weight_pruned)
    new_fc.bias = nn.Parameter(fc_bias_pruned)
    
    # Criar novo modelo com a mesma classe
    class PrunedSimpleModelWithFlashAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(
                new_layers[0],  # conv1
                new_layers[1],  # ReLU
                new_layers[2]   # MaxPool
            )
            self.conv2 = nn.Sequential(
                new_layers[3],  # conv2
                new_layers[4],  # ReLU
                new_layers[5]   # MaxPool
            )
            
            self.feature_dim = new_dim
            self.n_heads = n_heads
            self.head_dim = new_head_dim
            
            self.to_qkv = new_to_qkv
            self.attention_out = new_attention_out
            
            self.fc1 = nn.Sequential(
                new_fc1_linear,
                nn.ReLU(inplace=True)
            )
            self.fc = new_fc
            
        def forward(self, x):
            # Parte convolucional
            out = self.conv1(x)
            out = self.conv2(out)
            
            # Preparar para atenção
            out = out.flatten(2)
            out = out.transpose(1, 2)
            
            batch_size, seq_len, features = out.shape
            
            # Atenção
            qkv = self.to_qkv(out).reshape(
                batch_size, seq_len, 3, self.n_heads, self.head_dim
            )
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.2
            )
            
            attn_output = attn_output.transpose(1, 2).reshape(
                batch_size, seq_len, features
            )
            attn_output = self.attention_out(attn_output)
            
            # Pooling e MLP
            aggregated = attn_output.mean(dim=1)
            out = self.fc1(aggregated)
            out = self.fc(out)
            
            return out
    
    new_model = PrunedSimpleModelWithFlashAttention()
    
    # Calcular redução de parâmetros
    original_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in new_model.parameters())
    reduction = (1 - pruned_params / original_params) * 100
    
    print(f"Modelo reestruturado com sucesso!")
    print(f"Parâmetros originais: {original_params:,}")
    print(f"Parâmetros após pruning: {pruned_params:,}")
    print(f"Redução: {reduction:.2f}%")
    
    return new_model, masks