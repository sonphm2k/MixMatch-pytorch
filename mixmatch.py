import torch
import torch.nn.functional as F
# Configs
configs = {
    "epochs": 1,
    "batch_size": 2,
    "num_classes": 10,
    # -----------------
    "lr": 10e-6,
    "optim": "ADAM",  # type of optimizer [Adam, SGD]
    "lambda_u": 30,
    # 75       Hyper-parameter weighting the contribution of the unlabeled examples to the training loss
    "alpha": 0.75,  # 0.75     Hyperparameter for the Beta distribution used in MixU
    "T": 0.5,  # 0.5      Temperature parameter for sharpening used in MixMatch
    "K": 3,  # 3        Number of augmentations used when guessing labels in MixMatch
    "ema_decay": 0.999,
}

def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_model, train_criterion, epoch, device, config):
    model.train()
    list_losses = []
    list_losses_x = []
    list_losses_u = []
    # configs
    num_classes = configs['num_classes']
    batch_size = configs['batch_size']
    lambda_u = configs['lambda_u']
    alpha = configs['alpha']
    T = configs['T']
    K = configs['K']

    train_iteration = len(unlabeled_trainloader) + 1 # all_test_img / batch_size + 1
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    for batch_idx in range(train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)
        try:
            _, (inputs_u1, inputs_u2, inputs_u3) = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            _, (inputs_u1, inputs_u2, inputs_u3) = next(unlabeled_train_iter)

        inputs_x = inputs_x.to(device)
        inputs_u1 = inputs_u1.to(device)
        inputs_u2 = inputs_u2.to(device)
        inputs_u3 = inputs_u3.to(device)
        # Target -> Logits
        targets_x_new = torch.zeros(inputs_x.size(0), num_classes).scatter_(1, targets_x.view(-1,1).long(), 1)
        targets_x_new = targets_x_new.to(device)
        with torch.no_grad():
            outputs_u1 = model(inputs_u1)
            outputs_u2 = model(inputs_u2)
            outputs_u3 = model(inputs_u3)
            p = (torch.softmax(outputs_u1, dim=1) + torch.softmax(outputs_u2, dim=1) + torch.softmax(outputs_u3, dim=1)) / K
            # Sharpening - Hyper param
            targets_u = sharpen(p, T)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u1, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x_new, targets_u, targets_u, targets_u], dim=0)
        mixed_input, mixed_target = mixup(batch_size, all_inputs, all_targets, alpha)
        # logits
        logits = []
        for input in mixed_input:
            logits.append(model(input))
         # put interleaved samples back
        logits = mixmatch_interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        Lx, Lu, w = train_criterion(logits_x, mixed_target[:batch_size],
                                    logits_u, mixed_target[batch_size:],
                                    epoch, batch_idx, train_iteration, lambda_u)
        loss = Lx + w * Lu
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_model.update_params()
        # record loss
        list_losses.append(loss.item())
        list_losses_x.append(Lx.item())
        list_losses_u.append(Lu.item())

    losses = sum(list_losses) / len(list_losses)
    losses_x = sum(list_losses_x) / len(list_losses)
    losses_u = sum(list_losses_u) / len(list_losses)

    return losses, losses_x, losses_u

def sharpen(p, T):
    pt = p**(1/T)
    return pt / pt.sum(dim=1, keepdim=True)

def mixup(batch_size, all_inputs, all_targets, alpha):
    beta_dist = torch.distributions.beta.Beta(alpha, alpha)
    l = beta_dist.sample().item()
    l = max(l, 1-l)
    idx = torch.randperm(all_inputs.size(0))
    input_a, input_b = all_inputs, all_inputs[idx]
    target_a, target_b = all_targets, all_targets[idx]
    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b
    # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
    mixed_input = list(torch.split(mixed_input, batch_size))
    mixed_input = mixmatch_interleave(mixed_input, batch_size)
    return mixed_input, mixed_target

def mixmatch_interleave(xy, batch):
    nu = len(xy) - 1
    offsets = mixmatch_interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def mixmatch_interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

class SemiLoss(object):
    def __call__(self, logits_x, targets_x, logits_u, targets_u, epoch, batch_idx, train_iteration, lambda_u):
        Lx =  -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * targets_x, dim=1)) # Cross entropy

        probs_u = torch.softmax(logits_u, dim=1) # score of u
        Lu = F.mse_loss(probs_u, targets_u)   # MSE loss

        linear_weight = float(torch.clip(torch.tensor(epoch+(batch_idx/train_iteration)), min=0.0, max=1.0)) # range(0,1)
        w = lambda_u * linear_weight # range(0, lambda_u)
        return Lx, Lu, w

