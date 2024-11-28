from torch.utils.data import Dataset,DataLoader
import torch
import wandb
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc

device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'

def train_TopkMMoE(mymodel, train_dataset, val_dataset, bestmodel_save_dir, lr, N_epochs, batch_size):
    print(f"Currently using device:{device}")
    tot_iters = 0
    mymodel = mymodel.to(device)
    loss_fun = nn.BCELoss()
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr)
    adam_batch_loss = []
    losses = []
    val_losses = []
    best_loss = float("inf")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(N_epochs):
        # train loop
        batch_loss = []
        mymodel.train()
        for x_d_batch, x_c_batch, y1_batch, y2_batch in train_dataloader:
            tot_iters += 1

            x_d_batch = x_d_batch.to(device)
            x_c_batch = x_c_batch.to(device)
            y1_batch = y1_batch.to(device)
            y2_batch = y2_batch.to(device)

            [y1_pred, y2_pred], load_balancing_loss, router_z_loss = mymodel(x_d_batch, x_c_batch)  # 两个task
            y1_pred = y1_pred.squeeze(1)
            y2_pred = y2_pred.squeeze(1)

            loss1 = loss_fun(y1_pred, y1_batch)
            loss2 = loss_fun(y2_pred, y2_batch)
            wandb.log({"task1_loss": loss1, "task2_loss": loss2, "load_balancing_loss": load_balancing_loss,
                       "router_z_loss": router_z_loss})

            loss = loss1 + loss2 + load_balancing_loss + router_z_loss  # 此处令两个任务的损失值权重均为1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record result
            adam_batch_loss.append(loss.detach().cpu().numpy())
            batch_loss.append(loss.detach().cpu().numpy())

        # val loop
        val_batch_loss = []
        mymodel.eval()
        for x_d_batch, x_c_batch, y1_batch, y2_batch in val_dataloader:
            x_d_batch = x_d_batch.to(device)
            x_c_batch = x_c_batch.to(device)
            y1_batch = y1_batch.to(device)
            y2_batch = y2_batch.to(device)

            [y1_pred, y2_pred], load_balancing_loss, router_z_loss = mymodel(x_d_batch, x_c_batch)  # 两个task
            y1_pred = y1_pred.squeeze(1)
            y2_pred = y2_pred.squeeze(1)

            loss = loss_fun(y1_pred, y1_batch) + loss_fun(y2_pred, y2_batch) + load_balancing_loss + router_z_loss

            # record result
            val_batch_loss.append(loss.detach().cpu().numpy())

        # post processing
        losses.append(np.mean(np.array(batch_loss)))
        val_losses.append(np.mean(np.array(val_batch_loss)))

        # print progress
        print(f"Epoch={epoch},train_loss={losses[-1]},val_loss={val_losses[-1]}")
        wandb.log({"train_loss": losses[-1], "val_loss": val_losses[-1]})

        # save best model
        if (val_losses[-1] < best_loss):
            print("current epoch is the best so far. Saving model...")
            torch.save(mymodel.state_dict(), bestmodel_save_dir)
            best_loss = val_losses[-1]

    return losses, val_losses, adam_batch_loss

def get_auc(y_true,y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    calculated_auc = auc(fpr, tpr)
    print(f"AUC: {calculated_auc}")
    return calculated_auc

def eval_TopkMMoE(mybestmodel, test_dataset):
    mybestmodel = mybestmodel.to(device)
    mybestmodel.eval()
    x_test_d, x_test_c = test_dataset.x_d.to(device), test_dataset.x_c.to(device)
    [y1_pred, y2_pred], load_balancing_loss, router_z_loss = mybestmodel(x_test_d, x_test_c)
    y1_pred = y1_pred.squeeze(1).detach().cpu().numpy()
    y2_pred = y2_pred.squeeze(1).detach().cpu().numpy()
    auc1 = get_auc(test_dataset.y1, y1_pred)
    auc2 = get_auc(test_dataset.y2, y2_pred)
    wandb.log({"task1_AUC": auc1, "task2_AUC": auc2})
    return auc1, auc2

def train_MMoE(mymodel, train_dataset, val_dataset, bestmodel_save_dir, lr, N_epochs, batch_size):
    tot_iters = 0
    mymodel = mymodel.to(device)
    loss_fun = nn.BCELoss()
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr)
    adam_batch_loss = []
    losses = []
    val_losses = []
    best_loss = float("inf")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(N_epochs):
        # train loop
        batch_loss = []
        mymodel.train()
        for x_d_batch, x_c_batch, y1_batch, y2_batch in train_dataloader:
            tot_iters += 1

            x_d_batch = x_d_batch.to(device)
            x_c_batch = x_c_batch.to(device)
            y1_batch = y1_batch.to(device)
            y2_batch = y2_batch.to(device)

            y1_pred, y2_pred = mymodel(x_d_batch, x_c_batch)  # 两个task
            y1_pred = y1_pred.squeeze(1)
            y2_pred = y2_pred.squeeze(1)

            loss1 = loss_fun(y1_pred, y1_batch)
            loss2 = loss_fun(y2_pred, y2_batch)
            wandb.log({"task1_loss": loss1, "task2_loss": loss2})
            loss = loss1 + loss2  # 此处令两个任务的损失值权重均为1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record result
            adam_batch_loss.append(loss.detach().cpu().numpy())
            batch_loss.append(loss.detach().cpu().numpy())

        # val loop
        val_batch_loss = []
        mymodel.eval()
        for x_d_batch, x_c_batch, y1_batch, y2_batch in val_dataloader:
            x_d_batch = x_d_batch.to(device)
            x_c_batch = x_c_batch.to(device)
            y1_batch = y1_batch.to(device)
            y2_batch = y2_batch.to(device)

            y1_pred, y2_pred = mymodel(x_d_batch, x_c_batch)  # 两个task
            y1_pred = y1_pred.squeeze(1)
            y2_pred = y2_pred.squeeze(1)

            loss = loss_fun(y1_pred, y1_batch) + loss_fun(y2_pred, y2_batch)

            # record result
            val_batch_loss.append(loss.detach().cpu().numpy())

        # post processing
        losses.append(np.mean(np.array(batch_loss)))
        val_losses.append(np.mean(np.array(val_batch_loss)))

        # print progress
        print(f"Epoch={epoch},train_loss={losses[-1]},val_loss={val_losses[-1]}")
        wandb.log({"train_loss": losses[-1], "val_loss": val_losses[-1]})

        # save best model
        if (val_losses[-1] < best_loss):
            print("current epoch is the best so far. Saving model...")
            torch.save(mymodel.state_dict(), bestmodel_save_dir)
            best_loss = val_losses[-1]

    return losses, val_losses, adam_batch_loss

def eval_MMoE(mybestmodel, test_dataset):
    mybestmodel = mybestmodel.to(device)
    mybestmodel.eval()
    x_test_d, x_test_c = test_dataset.x_d.to(device), test_dataset.x_c.to(device)
    y1_pred, y2_pred = mybestmodel(x_test_d, x_test_c)
    y1_pred = y1_pred.squeeze(1).detach().cpu().numpy()
    y2_pred = y2_pred.squeeze(1).detach().cpu().numpy()
    auc1 = get_auc(test_dataset.y1, y1_pred)
    auc2 = get_auc(test_dataset.y2, y2_pred)
    wandb.log({"task1_AUC": auc1, "task2_AUC": auc2})
    return auc1, auc2