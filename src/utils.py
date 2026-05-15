import math
import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
import torch

def inv_sigmoid(x):
    return torch.log(x / (1 - x))

def normal_pdf(x, mu=0.0, sigma=1.0):
    norm_const = 1 / torch.sqrt(torch.tensor(2.0 * torch.pi, device=x.device, dtype=x.dtype) * sigma ** 2)
    exp_quad_term = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return norm_const * exp_quad_term

def log_normal_pdf(x, mu=0.0, sigma=1.0):
    log_norm_const = -math.log(math.sqrt(2.0 * math.pi * sigma**2))
    quad_term = -0.5 * ((x - mu) / sigma) ** 2
    return log_norm_const + quad_term

def read_npz(path, key='arr_0', dtype=torch.float32):
    data = np.load(path)
    return torch.as_tensor(data[key], dtype=dtype)

def pad_image(image):
    D, C, H, W = image.shape
    size = max(H, W)
    pad_val = image.min()
    padded_image = torch.full((D, C, size, size), pad_val, dtype=image.dtype, device=image.device)
    padded_image[:,:,(size-H)//2:(size-H)//2+H,(size-W)//2:(size-W)//2+W] = image
    return padded_image

def collate_fn(batch):
    images, lengths, labels = zip(*batch)
    #images, lengths, labels, lengths_labels = zip(*batch)
    images = torch.cat(images)
    labels = torch.stack(labels)
    return images, lengths, labels
    #return images, lengths, labels, lengths_labels
    
def instance_level_collate_fn(batch):
    images, lengths, labels = zip(*batch)
    images = torch.cat(images)
    labels = torch.cat(labels)
    return images, lengths, labels

def forward_with_lengths(self, x, lengths):
    x_split = torch.split(x, lengths)
    if len(set(x_i.shape[0] for x_i in x_split)) == 1:
        out = self.original_forward(torch.stack(x_split).permute(0, 2, 1, 3, 4))
    else:
        out = torch.cat([self.original_forward(x_i.permute(1, 0, 2, 3).unsqueeze(0)) for x_i in x_split])
    return out, None

def multiple_modalities_forward(self, x, lengths):
    stacked_x_split = torch.stack(torch.split(x, lengths))
    B, S, C, W, H = stacked_x_split.shape
    out = self.original_forward(stacked_x_split.reshape(B * S * C, W, H).unsqueeze(dim=1))
    return out.reshape(B * S, -1)
    
def encode_image(model, image):
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()

    with torch.no_grad():
        
        if device.type == 'cuda':
            image = image.to(device)
            
        encoded_image = model(image)
        
        if device.type == 'cuda':
            encoded_image = encoded_image.cpu()
            
    return encoded_image

def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler=None):

    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.train()
        
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'auroc': 0.0, 'auprc': 0.0, 'bal_acc': 0.0, 'labels': [], 'logits': [], 'loss': 0.0, 'nll': 0.0}

    for images, lengths, labels in dataloader:
    #for images, lengths, labels, lengths_labels in dataloader:
                
        batch_size = len(lengths)

        if device.type == 'cuda':
            images = images.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()

        params = torch.nn.utils.parameters_to_vector(model.parameters())
        logits, attn_weights = model(images, lengths)
        losses = criterion(logits, labels, attn_weights=attn_weights, lengths=lengths, params=params, n=len(dataloader.dataset))
        #losses = criterion(logits, labels, attn_weights=attn_weights, lengths=lengths, lengths_labels=lengths_labels, params=params, n=len(dataloader.dataset))
        losses['loss'].backward()

        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], max_norm=1.0)

        optimizer.step()

        if lr_scheduler:
            lr_scheduler.step()

        metrics['loss'] += (batch_size / dataset_size) * losses['loss'].item()
        metrics['nll'] += (batch_size / dataset_size) * losses['nll'].item()

        if device.type == 'cuda':
            labels = labels.detach().cpu()
            logits = logits.detach().cpu()

        metrics['labels'].extend(labels)
        metrics['logits'].extend(logits)
            
    logits = torch.stack(metrics['logits'])
    probs = torch.nn.functional.sigmoid(logits).detach().numpy()
    preds = (probs >= 0.5).astype(int)
    labels = torch.stack(metrics['labels'])
    metrics['auroc'] = roc_auc_score(labels.numpy(), probs)
    metrics['auprc'] = average_precision_score(labels.numpy(), probs)
    metrics['bal_acc'] = balanced_accuracy_score(labels.numpy(), preds)
    
    return metrics

def evaluate(model, criterion, dataloader):

    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()
        
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'auroc': 0.0, 'auprc': 0.0, 'bal_acc': 0.0, 'labels': [], 'logits': [], 'loss': 0.0, 'nll': 0.0}

    with torch.no_grad():
        for images, lengths, labels in dataloader:
        #for images, lengths, labels, lengths_labels in dataloader:
            
            batch_size = len(lengths)

            if device.type == 'cuda':
                images, labels = images.to(device), labels.to(device)

            params = torch.nn.utils.parameters_to_vector(model.parameters())
            logits, attn_weights = model(images, lengths)
            losses = criterion(logits, labels, attn_weights=attn_weights, lengths=lengths, params=params, n=len(dataloader.dataset))
            #losses = criterion(logits, labels, attn_weights=attn_weights, lengths=lengths, lengths_labels=lengths_labels, params=params, n=len(dataloader.dataset))

            metrics['loss'] += (batch_size / dataset_size) * losses['loss'].item()
            metrics['nll'] += (batch_size / dataset_size) * losses['nll'].item()

            if device.type == 'cuda':
                labels, logits = labels.detach().cpu(), logits.detach().cpu()

            metrics['labels'].extend(labels)
            metrics['logits'].extend(logits)

        logits = torch.stack(metrics['logits'])
        probs = torch.nn.functional.sigmoid(logits).detach().numpy()
        preds = (probs >= 0.5).astype(int)
        labels = torch.stack(metrics['labels'])
        metrics['auroc'] = roc_auc_score(labels.numpy(), probs)
        metrics['auprc'] = average_precision_score(labels.numpy(), probs)
        metrics['bal_acc'] = balanced_accuracy_score(labels.numpy(), preds)
            
    return metrics
