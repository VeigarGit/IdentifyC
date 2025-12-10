import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import numpy as np


import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import numpy as np

def prune_and_restructure(model, pruning_rate = 0.5, n_in = 3, size_fc = 25, data='Cifar100'):
    model_copy = copy.deepcopy(model)
    layers = []
    indices_not_remove_weight = []
    first_module = True
    first_linear = True
    first_attention = True
    index_last_layer = 0
    masks = []
    a = 0
    for n, module in enumerate(model_copy.modules()):
        if isinstance(module, nn.Linear):
            index_last_layer = n

    for n, module in enumerate(model_copy.modules()):
        print(module)
        if isinstance(module, nn.Conv2d):
            
            # realiza o prunning e substitui os pesos por 0
            prune.ln_structured(module, name='weight', amount=pruning_rate, n=2, dim=0)
            masks.append(list(model_copy.buffers())[0])
            prune.remove(module, 'weight')

            # filtra onde não é 0
            indices_not_remove_perceptron = module.weight.abs().sum(dim=(1, 2, 3)) != 0

            # aplica o filtragem dos valores 0
            weight_prunned = module.weight[indices_not_remove_perceptron]

            # filtra os pesos da camada
            if first_module:
                first_module = False
            else:
                weight_prunned = weight_prunned[:, indices_not_remove_weight, :, :]
                
            if module.bias is not None:
                bias_prunned = module.bias[indices_not_remove_perceptron]
            
            # determina tamanho da saída
            n_out = sum(indices_not_remove_perceptron)

            # cria nova cadamada com os parametros filtrados
            new_layer = nn.Conv2d(
                        in_channels=n_in,
                        out_channels=n_out,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        bias=(module.bias is not None)
                        )
            new_layer.bias = nn.Parameter(bias_prunned)
            new_layer.weight = nn.Parameter(weight_prunned)
            
            layers.append(new_layer)
            indices_not_remove_weight = indices_not_remove_perceptron
            n_in = n_out
            
        if isinstance(module, nn.Linear) and hasattr(module, 'is_attention'):
            if first_attention:
                new_layer = nn.Linear(128, 128 * 3)
                first_attention = False
            else:
                new_layer = nn.Linear(128, 128)
            new_layer.bias = module.bias
            new_layer.weight = module.weight
            print('Atenção adicionada')
            layers.append(new_layer)

        if isinstance(module, nn.Linear) and not hasattr(module, 'is_attention'):            
            
            # realiza o prunning e substitui os pesos por 0
            prune.ln_structured(module, name='weight', amount=pruning_rate, n=2, dim=0)
            masks.append(list(model_copy.buffers())[0])
            prune.remove(module, 'weight')

            # filtra onde não é 0
            indices_not_remove_perceptron = module.weight.abs().sum(dim=1) != 0

            # filtra os pesos da camada
            if first_linear:
                if data=='Cifar100' or data=='Cifar10':
                    print('Usou o cifar')
                    indices_not_remove_weight = indices_not_remove_weight.repeat(25)
                if data=='MNIST':
                    print('Usou o mnist')
                    indices_not_remove_weight = indices_not_remove_weight.repeat(16)
                if data=='cancer':
                    print('Usou cancer')
                    
                first_linear = False
                
                n_in = sum(indices_not_remove_weight)
                layers.append(nn.Flatten())
            
            # remove os pesos que não serão utilizados
            weight_prunned = module.weight[:, indices_not_remove_weight]
            # aplica o filtragem nos neuronios com valores 0
            if data=='Cifar100':
                ind=100
            elif data=='Cifar10':
                ind=10
            elif data=='cancer':
                ind=26
            if n == index_last_layer: # não reestrutura a ultima camada
                indices_not_remove_perceptron = torch.tensor([True] * ind)
                masks[-1] = masks[-1] + 1
            
            weight_prunned = weight_prunned[indices_not_remove_perceptron]
                
            if module.bias is not None:
                bias_prunned = module.bias[indices_not_remove_perceptron]
                
            # determina tamanho da saída
            n_out = sum(indices_not_remove_perceptron)

            # cria nova cadamada com os parametros filtrados
            new_layer = nn.Linear(n_in, n_out)
            new_layer.bias = nn.Parameter(bias_prunned)
            new_layer.weight = nn.Parameter(weight_prunned)
            
            layers.append(new_layer)
            indices_not_remove_weight = indices_not_remove_perceptron
            n_in = n_out
        
        if isinstance(module, (nn.ReLU, nn.MaxPool2d)):
            #adiciona as camadas de funções de ativação 
            layers.append(module)
            print('Ativação adicionada')
            
        
        new_model = nn.Sequential(*layers)
    print("Modelo reestruturado com sucesso.")
    print(new_model)
    print("rapaz ne que foi")
    return new_model, masks