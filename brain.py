import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ========== CONFIGURAÇÃO DOS DIRETÓRIOS DOS DADOS ==========
# Diretório raiz do dataset Multi Cancer
IMAGE_DIR = './data/Multi Cancer/Multi Cancer'

# Diretórios para treino e teste
TRAIN_DIR = './train_data_multicancer'
TEST_DIR = './test_data_multicancer'

# ========== LISTA DE TODAS AS CLASSES ==========
# Baseado na estrutura fornecida
ALL_CLASSES = {
    # Blood Cancer
    'all_benign': 'Blood Cancer/Benign',
    'all_early': 'Blood Cancer/Early Leukemia',
    'all_pre': 'Blood Cancer/Pre Leukemia',
    'all_pro': 'Blood Cancer/Pro Leukemia',
    
    # Brain Cancer
    'brain_glioma': 'Brain Cancer/Glioma',
    'brain_menin': 'Brain Cancer/Meningioma',
    'brain_tumor': 'Brain Cancer/Pituitary Tumor',
    
    # Breast Cancer
    'breast_benign': 'Breast Cancer/Benign',
    'breast_malignant': 'Breast Cancer/Malignant',
    
    # Cervix Cancer
    'cervix_dyk': 'Cervical Cancer/cervix_dyk',
    'cervix_koc': 'Cervical Cancer/cervix_koc',
    'cervix_mep': 'Cervical Cancer/cervix_mep',
    'cervix_pab': 'Cervical Cancer/cervix_pab',
    'cervix_sfi': 'Cervical Cancer/cervix_sfi',
    
    # Kidney Cancer
    'kidney_normal': 'Kidney Cancer/kidney_normal',
    'kidney_tumor': 'Kidney Cancer/kidney_tumor',
    
    # Colon and Lung Cancer
    'colon_aca': 'Colon-Lung Cancer/colon_aca',
    'colon_bnt': 'Colon-Lung Cancer/colon_bnt',
    'lung_aca': 'Colon-Lung Cancer/lung_aca',
    'lung_bnt': 'Colon-Lung Cancer/lung_bnt',
    'lung_scc': 'Colon-Lung Cancer/lung_scc',
    
    # Lymph Cancer
    'lymph_cll': 'Lymph Cancer/lymph_cll',
    'lymph_fl': 'Lymph Cancer/lymph_fl',
    'lymph_mcl': 'Lymph Cancer/lymph_mcl',
    
    # Oral Cancer
    'oral_normal': 'Oral Cancer/oral_norma',
    'oral_scc': 'Oral Cancer/oral_scc'
}

# ========== FUNÇÃO PARA PREPARAR OS DADOS ==========
def prepare_data():
    """Prepara os dados dividindo em treino e teste"""
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    def split_and_copy(class_key, class_name, test_size=0.2, seed=42):
        # Encontrar o caminho correto baseado no nome da classe
        # Primeiro, vamos procurar a classe recursivamente
        class_dir = None
        for root, dirs, files in os.walk(IMAGE_DIR):
            for dir_name in dirs:
                if dir_name == class_key:
                    class_dir = os.path.join(root, dir_name)
                    break
            if class_dir:
                break
        
        if not class_dir:
            print(f"Aviso: Diretório não encontrado para classe {class_key}")
            return
        
        # Listar todas as imagens
        images = [
            os.path.join(class_dir, img)
            for img in os.listdir(class_dir)
            if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ]
        
        if len(images) == 0:
            print(f"Aviso: Nenhuma imagem encontrada para classe {class_key}")
            return
        
        # Dividir em treino e teste
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=seed)
        
        # Criar diretórios de saída
        train_output_dir = os.path.join(TRAIN_DIR, class_key)
        test_output_dir = os.path.join(TEST_DIR, class_key)
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Copiar imagens para treino
        for img in train_images:
            dest_path = os.path.join(train_output_dir, os.path.basename(img))
            if not os.path.exists(dest_path):
                copyfile(img, dest_path)
        
        # Copiar imagens para teste
        for img in test_images:
            dest_path = os.path.join(test_output_dir, os.path.basename(img))
            if not os.path.exists(dest_path):
                copyfile(img, dest_path)
        
        print(f"  {class_name}: {len(train_images)} treino, {len(test_images)} teste")
    
    print("Preparando dados para todas as classes de câncer...")
    
    # Processar cada classe
    for class_key, class_name in ALL_CLASSES.items():
        split_and_copy(class_key, class_name)

# ========== MODELO SIMPLES CORRIGIDO ==========
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

def evaluate_model(model, test_loader, device, class_names):
    """Avaliação detalhada do modelo"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Matriz de confusão
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    # Relatório de classificação
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return all_preds, all_labels

# ========== FUNÇÃO PRINCIPAL ==========
def main():
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    # Preparar dados (se necessário)
    if not os.path.exists(TRAIN_DIR):
        print("Preparando dados...")
        prepare_data()
    
    # Hiperparâmetros
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 5
    val_ratio = 0.2
    
    # Transformações para treino (com aumento de dados)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Carregar datasets
    print("Carregando datasets...")
    full_train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform_train)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform_test)
    
    # Nomes das classes
    class_names = full_train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Número total de imagens de treino: {len(full_train_dataset)}")
    print(f"Número total de imagens de teste: {len(test_dataset)}")
    
    # Dividir treino em treino/validação
    val_size = int(val_ratio * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    print(f"Tamanho do treino: {train_size}, Tamanho da validação: {val_size}")
    
    # Criar DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2)
    
    # Criar modelo
    print("Criando modelo...")
    model = SimpleModelWithFlashAttention(
        in_features=3,
        num_classes=len(class_names)
    ).to(device)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")
    
    # Verificar se a dimensão está correta
    # Testar com um tensor de entrada
    test_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(test_input)
    print(f"Dimensão de saída do modelo: {output.shape}")
    
    # Definir otimizador, loss e scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Listas para armazenar métricas
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    # Loop de treinamento
    print("Iniciando treinamento...")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Treinar
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validar
        val_loss, val_acc = test_epoch(model, val_loader, criterion, device)
        
        # Ajustar learning rate
        scheduler.step()
        
        # Salvar melhores pesos
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'best_model_brain_cancer.pth')
            print(f"  Melhor modelo salvo! Acc: {val_acc:.2f}%")
        
        # Armazenar métricas
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)
    
    # Salvar modelo final
    torch.save(model.state_dict(), 'final_model_brain_cancer.pth')
    print(f"\nTreinamento concluído! Melhor val acc: {best_val_acc:.2f}%")
    
    # ========== VISUALIZAÇÃO DOS RESULTADOS ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfico de loss
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss durante o treinamento')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico de acurácia
    axes[1].plot(train_accs, label='Train Accuracy', linewidth=2)
    axes[1].plot(val_accs, label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Acurácia durante o treinamento')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results_brain_cancer.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ========== TESTAR NO CONJUNTO DE TESTE ==========
    print("\nAvaliando no conjunto de teste...")
    
    # Carregar melhor modelo
    checkpoint = torch.load('best_model_brain_cancer.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Avaliação detalhada
    all_preds, all_labels = evaluate_model(model, test_loader, device, class_names)
    
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
    for i in range(min(8, len(images))):
        pred = class_names[predicted[i]]
        actual = class_names[labels[i]]
        correct = "✓" if predicted[i] == labels[i] else "✗"
        print(f"  Imagem {i+1}: Predição={pred}, Real={actual} {correct}")

# ========== EXECUTAR ==========
if __name__ == "__main__":
    main()