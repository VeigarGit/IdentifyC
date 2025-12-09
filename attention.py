import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from flops import count_flops
# ========== DEFINIÇÃO DO MODELO COM FLASH ATTENTION ==========
class SimpleModelWithFlashAttention(nn.Module):
    def __init__(self, in_features=3, num_classes=10, dim=128, n_heads=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, dim, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # AGORA: dim = 64 (canais), NÃO 1600!
        self.feature_dim = dim  # 64 canais
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, "dim deve ser divisível por n_heads"
        
        # Projeções para Q, K, V - atenção à dimensão de entrada!
        self.to_qkv = nn.Linear(dim, dim * 3)  # Entrada: 64 (features por posição)
        self.attention_out = nn.Linear(dim, dim)
        
        # AGORA: entrada do fc1 depende de como agregamos as 25 posições
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),  # Se fizermos pooling sobre as posições
            # OU: nn.Linear(dim * 25, 512) se concatenarmos tudo
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Parte convolucional (igual)
        out = self.conv1(x)  # (batch, 32, 14, 14)
        out = self.conv2(out)  # (batch, 64, 5, 5)
        
        # MUDANÇA CRÍTICA AQUI:
        # Não fazer flatten total! Queremos manter as posições espaciais
        # Transformar (batch, 64, 5, 5) → (batch, 25, 64)
        
        # 1. Flatten apenas as dimensões espaciais
        out = out.flatten(2)  # (batch, 64, 25)
        # 2. Transpor para ter sequência de 25 elementos com 64 features cada
        out = out.transpose(1, 2)  # (batch, 25, 64)
        
        batch_size, seq_len, features = out.shape  # seq_len = 25
        
        # Criar Q, K, V - AGORA para 25 elementos!
        qkv = self.to_qkv(out).reshape(
            batch_size, seq_len, 3, self.n_heads, self.head_dim
        )
        # Reorganizar dimensões para [3, batch, heads, seq_len, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Flash Attention - AGORA com sequência real!
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.5
        )
        
        # Reformatar: (batch, heads, seq_len, head_dim) → (batch, seq_len, features)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, features
        )
        
        # Projeção de saída do attention
        attn_output = self.attention_out(attn_output)
        
        # MUDANÇA: Como agregar as 25 posições?
        # Opção 1: Pooling global (média sobre as posições)
        aggregated = attn_output.mean(dim=1)  # (batch, 64)
        # Opção 2: Usar só a primeira posição (como [CLS] token)
        # aggregated = attn_output[:, 0, :]  # (batch, 64)
        # Opção 3: Concatenar tudo
        # aggregated = attn_output.flatten(1)  # (batch, 64*25=1600)
        
        # Continue com as camadas fully connected
        out = self.fc1(aggregated)  # (batch, 512)
        out = self.fc(out)  # (batch, num_classes)
        return out

# ========== FUNÇÕES DE TREINAMENTO E AVALIAÇÃO ==========
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'Loss': running_loss/(batch_idx+1),
            'Acc': 100.*correct/total
        })
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def test_epoch(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

# ========== FUNÇÃO PRINCIPAL ==========
def main():
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    # Verificar se Flash Attention está disponível
    print(f"PyTorch version: {torch.__version__}")
    print(f"Flash Attention disponível: {hasattr(F, 'scaled_dot_product_attention')}")
    
    # Habilitar Flash Attention se disponível
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        print("Flash Attention habilitado")
    
    # Hiperparâmetros
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 5
    
    # Transformações dos dados
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Carregar datasets
    print("Carregando CIFAR-10...")
    train_dataset = datasets.CIFAR100(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR100(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    # Classes do CIFAR-10
    classes = (
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
)
    
    # Criar modelo
    print("Criando modelo...")
    model = SimpleModelWithFlashAttention(
        in_features=3,
        num_classes=100,
    ).to(device)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_flops1, breakdown1, gflops = count_flops(model, verbose=True)
    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")
    print(f"model Flops: {gflops:.2f} GFLOPs")
    # Definir otimizador e loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Listas para armazenar métricas
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    # Loop de treinamento
    print("Iniciando treinamento...")
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Treinar
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Testar
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        
        # Ajustar learning rate
        scheduler.step()
        
        # Salvar melhores pesos
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model_flash_attention.pth')
        
        # Armazenar métricas
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        print(f"  Best Acc:   {best_acc:.2f}%")
        print("-" * 50)
    
    # Salvar modelo final
    torch.save(model.state_dict(), 'final_model_flash_attention.pth')
    print(f"\nTreinamento concluído! Melhor acurácia: {best_acc:.2f}%")
    
    # ========== VISUALIZAÇÃO DOS RESULTADOS ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfico de loss
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(test_losses, label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss durante o treinamento')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico de acurácia
    axes[1].plot(train_accs, label='Train Accuracy', linewidth=2)
    axes[1].plot(test_accs, label='Test Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Acurácia durante o treinamento')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results_flash_attention.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ========== TESTAR COM IMAGENS EXEMPLO ==========
    print("\nTestando com algumas imagens de exemplo...")
    model.eval()
    
    # Pegar um batch de teste
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images[:8].to(device), labels[:8].to(device)
    
    # Fazer predições
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Mostrar resultados
    print("Predições:")
    for i in range(8):
        pred = classes[predicted[i]]
        actual = classes[labels[i]]
        correct = "✓" if predicted[i] == labels[i] else "✗"
        print(f"  Imagem {i+1}: Predição={pred}, Real={actual} {correct}")

# ========== EXECUTAR ==========
if __name__ == "__main__":
    main()
