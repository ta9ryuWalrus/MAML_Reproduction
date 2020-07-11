import torch
from collections import OrderedDict

def adaptation(model, optimizer, batch, loss_fn, lr, train_step, train, device):
    # losses = []
    predictions = []
    labels = []
    x_train, y_train = batch['train']
    x_val, y_val = batch['test']

    for idx in range(x_train.size(0)):
        weights = OrderedDict(model.named_parameters())

        # k-shotでadaptation
        input_x = x_train[idx].to(device)
        input_y = y_train[idx].to(device)

        optimizer.zero_grad()
        for iter in range(train_step):
            logits = model.adaptation(input_x, weights)
            loss = loss_fn(logits, input_y)
            gradients = torch.autograd.grad(loss, weights.values(), create_graph=train)

            weights = OrderedDict((name, param - lr * grad) for ((name, param), grad) in zip(weights.items(), gradients))

        # queryで評価
        input_x = x_val[idx].to(device)
        input_y = y_val[idx].to(device)
        logits = model.adaptation(input_x, weights)
        model.train()
        loss = loss_fn(logits, input_y)
        loss.backward(retain_graph=True)
        if train:
            optimizer.step()

        y_pred = logits.softmax(dim=1)
        predictions.append(y_pred)
        labels.append(input_y)
        # losses.append(loss)

    # model.train()
    # optimizer.zero_grad()
    # batch_loss = torch.stack(losses).mean()
    '''
    if train:
        batch_loss.backward()
        optimizer.step()
    '''
    y_pred = torch.cat(predictions)
    y_label = torch.cat(labels)
    batch_acc = torch.eq(y_pred.argmax(dim=-1), y_label).sum().item() / y_pred.shape[0]
    return loss, batch_acc