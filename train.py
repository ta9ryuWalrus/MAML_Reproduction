import torch
from collections import OrderedDict

def adaptation(model, optimizer, batch, loss_fn, lr, train_step, train, device):
    losses = []
    predictions = []
    x_train, y_train = batch['train']
    x_val, y_val = batch['test']

    for idx in range(x_train.size(0)):
        weights = OrderedDict(model.named_parameters())

        # k-shotでadaptation
        input_x = x_train[idx].to(device)
        input_y = y_train[idx].to(device)

        for iter in range(train_step):
            logits = model.adaptation(input_x, weights)
            loss = loss_fn(logits, input_y)
            gradients = torch.autograd.grad(loss, weights.values(), create_graph=train)

            weights = OrderedDict((name, param - lr * grad) for ((name, param), grad) in zip(weights.items(), gradients))

        # queryで評価
        input_x = x_val[idx].to(device)
        input_y = y_val[idx].to(device)
        logits = model.adaptation(input_x, weights)
        loss = loss_fn(logits, input_y)
        loss.backward(retain_graph=True)

        y_pred = logits.softmax(dim=1)
        predictions.append(y_pred)
        losses.append(loss)

    model.train()
    optimizer.zero_grad()
    batch_loss = torch.stack(losses).mean()
    if train:
        batch_loss.backward()
        optimizer.step()
    y_pred = torch.cat(predictions)
    y_val = torch.cat(y_val)
    batch_acc = torch.eq(y_pred.argmax(dim=-1), y_val).sum().item() / y_pred.shape[0]
    return batch_loss, batch_acc