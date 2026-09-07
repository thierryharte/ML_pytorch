import torch
import os
import time
import logging

logger = logging.getLogger(__name__)

DEBUG = False


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def loop_one_batch(
    running_loss,
    tot_loss,
    running_correct,
    tot_correct,
    running_num,
    tot_num,
    count,
    i,
    data,
    model,
    optimizer,
    loss_fn,
    device,
    train,
    time_epoch,
    num_batches,
    epoch_index,
    eval_model,
    all_scores,
    all_labels,
    all_weights,
    all_kl_values,
    type_eval,
):
    inputs, labels, class_weights, kl_values = data

    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    class_weights = class_weights.to(device, non_blocking=True)
    kl_values = kl_values.to(device, non_blocking=True)

    event_weights = inputs[:, -1]
    inputs = inputs[:, :-1]
    weights = (event_weights * class_weights.flatten()).clamp(min=0)

    # compute the outputs of the model
    outputs = model(inputs)

    if outputs.shape[1] == 1:
        outputs = outputs.flatten()
        y_pred = torch.round(outputs)

    else:
        labels = labels.type(dtype=torch.long)
        y_pred = outputs.argmax(dim=1)

    # Compute the accuracy
    correct = ((y_pred == labels) * weights).sum().item()

    # Compute the loss and its gradients
    loss = loss_fn(outputs, labels)
    # reshape the loss
    loss = loss.view(1, -1).squeeze()
    # weight the loss
    loss = loss * weights
    # weighted average of the loss
    loss_average = loss.sum() / weights.sum()

    if train:
        # Reset the gradients of all optimized torch.Tensor
        optimizer.zero_grad()
        # Compute the gradients of the loss w.r.t. the model parameters
        loss_average.backward()
        # Adjust learning weights
        optimizer.step()

    # Gather data for reporting
    running_loss += loss_average.item()
    tot_loss += loss_average.item()

    running_correct += correct
    tot_correct += correct

    running_num += weights.sum().item()
    tot_num += weights.sum().item()

    step_prints = max(1, 0.1 * num_batches)

    if DEBUG:
        if loss_average.item() > 1:
            breakpoint()
        if not train:
            print("loss", loss)
        if not train:
            print("loss_average", loss_average)
        if not train:
            print("correct", correct)
        if not train:
            print("running_loss", running_loss)
        if not train:
            print("tot_loss", tot_loss)
        if not train:
            print("running_correct", running_correct)
        if not train:
            print("tot_correct", tot_correct)
        if not train:
            print("running_num", running_num)
        if not train:
            print("tot_num", tot_num)
        if not train:
            print("i", i)

    if i + 1 >= step_prints * count:
        if count == 2 and (epoch_index == 0 or eval_model):
            logger.info(f"outputs {outputs}, {outputs.shape}")
            logger.info(f"labels {labels}, {labels.shape}")
            logger.info(f"loss {loss}, {loss.shape}")
            logger.info(f"inputs {inputs}, {inputs.shape}")
            logger.info(f"weights {weights}, {weights.shape}")

        count += 1

        last_loss = running_loss / step_prints  # loss per batch
        last_accuracy = running_correct / running_num  # accuracy per batch
        tb_x = epoch_index * num_batches + i + 1

        if DEBUG:
            if not train:
                print("\n Prints")
            if not train:
                print("i", i)
            if not train:
                print("count", count)
            if not train:
                print("running_loss", running_loss)
            if not train:
                print("running_correct", running_correct)
            if not train:
                print("running_num", running_num)
            if not train:
                print("step_prints", step_prints)

        logger.info(
            "EPOCH # %d, time %.1f,  %s batch %.1f %% , %s        accuracy: %.4f      //      loss: %.4f"
            % (
                epoch_index,
                time.time() - time_epoch,
                (
                    ("Training" if train else "Validation")
                    if not eval_model
                    else f"Evaluating ({type_eval})"
                ),
                (i + 1) / num_batches * 100,
                f"step {tb_x}" if not eval_model else "",
                last_accuracy,
                last_loss,
            )
        )

        # type = "train" if train else "val"
        # tb_writer.add_scalar(f"Accuracy/{type}", last_accuracy, tb_x)
        # tb_writer.add_scalar(f"Loss/{type}", last_loss, tb_x)

        running_loss = 0.0
        running_correct = 0
        running_num = 0

    if eval_model:
        # Create array of scores and labels
        if i == 0:
            all_scores = outputs.cpu().detach()
            all_labels = labels.cpu().detach()
            all_weights = event_weights.cpu().detach()
            all_kl_values = kl_values.cpu().detach()
        else:
            all_scores = torch.cat((all_scores, outputs.cpu().detach()))
            all_labels = torch.cat((all_labels, labels.cpu().detach()))
            all_weights = torch.cat((all_weights, event_weights.cpu().detach()))
            all_kl_values = torch.cat((all_kl_values, kl_values.cpu().detach()))

    return (
        running_loss,
        running_correct,
        running_num,
        tot_loss,
        tot_correct,
        tot_num,
        count,
        all_scores,
        all_labels,
        all_weights,
        all_kl_values,
    )


def save_pytorch_model(main_dir, epoch_index, model, optimizer, save_model=False):
    model_dir = f"{main_dir}/models"
    state_dict_dir = f"{main_dir}/state_dict"
    save_model=False
    if save_model:
        os.makedirs(model_dir, exist_ok=True)
        model_name = f"{model_dir}/model_{epoch_index}.pt"
        torch.save(model, model_name)
    os.makedirs(state_dict_dir, exist_ok=True)
    state_dict_name = f"{state_dict_dir}/model_{epoch_index}_state_dict.pt"
    checkpoint = {
        "epoch": epoch_index,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, state_dict_name)
    return state_dict_name


def train_val_one_epoch(
    train,
    epoch_index,
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    time_epoch,
    scheduler,
    cfg,
    main_dir=None,
    best_loss=None,
    best_accuracy=None,
    best_epoch=None,
    best_model_name=None,
):
    logger.info("Training" if train else "Validation")
    if train:
        model.train(train)
    else:
        model.eval()

    running_loss = 0.0
    tot_loss = 0.0

    running_correct = 0
    tot_correct = 0

    running_num = 0
    tot_num = 0

    num_batches = len(loader)
    count = 0

    all_scores = None
    all_labels = None
    all_weights = None
    all_kl_values  = None

    # Loop over the training data
    for i, data in enumerate(loader):
        (
            running_loss,
            running_correct,
            running_num,
            tot_loss,
            tot_correct,
            tot_num,
            count,
            *_,
        ) = loop_one_batch(
            running_loss,
            tot_loss,
            running_correct,
            tot_correct,
            running_num,
            tot_num,
            count,
            i,
            data,
            model,
            optimizer,
            loss_fn,
            device,
            train,
            time_epoch,
            num_batches,
            epoch_index,
            False,
            all_scores,
            all_labels,
            all_weights,
            all_kl_values,
            None,
        )

    if train:
        logger.info(
            "EPOCH # %d, learning rate: %.6f"
            % (epoch_index, optimizer.param_groups[0]["lr"])
        )
        scheduler.step()

    avg_loss = tot_loss / len(loader)
    avg_accuracy = tot_correct / tot_num
    if not train:
        # Track best performance, and save the model state
        if cfg.eval_param == "loss":
            evaluator = avg_loss
            best_eval = best_loss
        elif cfg.eval_param == "acc":
            evaluator = 1 - avg_accuracy
            best_eval = 1 - best_accuracy
        else:
            raise ValueError("Bad evaluator name")

    if not train and evaluator < best_eval - cfg.min_delta:
        best_eval = evaluator
        best_loss = avg_loss
        best_accuracy = avg_accuracy
        best_epoch = epoch_index
        best_model_name = save_pytorch_model(main_dir, epoch_index, model, optimizer, cfg.save_model)

    return (
        avg_loss,
        avg_accuracy,
        best_loss,
        best_accuracy,
        best_epoch,
        best_model_name,
    )


def eval_model(model, loader, loss_fn, type_eval, device, best_epoch):
    # Test the model by running it on the test set
    running_loss = 0.0
    tot_loss = 0.0

    running_correct = 0
    tot_correct = 0

    running_num = 0
    tot_num = 0

    num_batches = len(loader)
    count = 0

    all_scores = None
    all_labels = None
    all_weights = None
    all_kl_values = None

    for i, data in enumerate(loader):
        (
            running_loss,
            running_correct,
            running_num,
            tot_loss,
            tot_correct,
            tot_num,
            count,
            all_scores,
            all_labels,
            all_weights,
            all_kl_values,
        ) = loop_one_batch(
            running_loss,
            tot_loss,
            running_correct,
            tot_correct,
            running_num,
            tot_num,
            count,
            i,
            data,
            model,
            None,
            loss_fn,
            device,
            False,
            time.time(),
            num_batches,
            best_epoch,
            True,
            all_scores,
            all_labels,
            all_weights,
            all_kl_values,
            type_eval,
        )

    avg_loss = tot_loss / len(loader)
    avg_accuracy = tot_correct / tot_num

    # Normalize raw logits into probabilities so that the saved scores live
    # in [0, 1] regardless of the model architecture.
    if torch.any(all_scores < 0) or torch.any(all_scores > 1):
        if all_scores.shape[1] == 1:
            all_scores = torch.nn.functional.sigmoid(all_scores)
        else:
            all_scores = torch.nn.functional.softmax(all_scores, dim=1)

    # For backward compatibility with the binary sig/bkg case (1 or 2 output
    # nodes) we collapse to a single column containing the signal probability.
    # For genuine multi-class outputs (>=3) we keep all class probabilities.
    if all_scores.shape[1] == 2:
        all_scores = all_scores[:, -1].view(-1, 1)
    elif all_scores.shape[1] == 1:
        all_scores = all_scores.view(-1, 1)

    all_labels = all_labels.view(-1, 1)
    all_weights = all_weights.view(-1, 1)
    all_kl_values = all_kl_values.view(-1, 1)

    score_lbl_tensor = torch.cat(
        (all_scores, all_labels, all_weights, all_kl_values), 1
    )
    logger.info(f"score_lbl_tensor shape: {score_lbl_tensor.shape}")

    # detach the tensor from the graph and convert to numpy array
    score_lbl_array = score_lbl_tensor.numpy()
    # score_lbl_array = score_lbl_tensor.cpu().detach().numpy()

    return score_lbl_array, avg_loss, avg_accuracy


def export_onnx(model, model_name, batch_size, input_size, device, onnx_model_name):
    model_dir = os.path.dirname(model_name)
    
    if hasattr(model, "export_model"):
        model = model.export_model(model)

    # Export the model to ONNX format
    dummy_input = torch.zeros(batch_size, input_size, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        f"{model_dir}/{onnx_model_name}.onnx",
        verbose=True,
        export_params=True,
        opset_version=13,
        input_names=["InputVariables"],  # the model's input names
        output_names=["Output"],  # the model's output names
        dynamic_axes={
            "InputVariables": {0: "batch_size"},  # variable length axes
            "Output": {0: "batch_size"},
        },
    )


def create_DNN_columns_list(run2, dnn_input_variables):
    """Create the columns of the DNN input variables"""
    column_list = []
    for x, y in dnn_input_variables.values():
        if run2:
            if ":" in x:
                coll, pos = x.split(":")
                column_list.append(f"{coll}Run2_{y}:{pos}")
            elif x != "events":
                column_list.append(f"{x}Run2_{y}")
            elif "sigma" in y:
                column_list.append(f"{x}_{y}Run2")
            else:
                column_list.append(f"{x}_{y}")
        else:
            if ":" in x:
                coll, pos = x.split(":")
                column_list.append(f"{coll}_{y}:{pos}")
            else:
                column_list.append(f"{x}_{y}")

    return column_list


if __name__ == "__main__":
    from ml_pytorch.defaults.dnn_input_variables import bkg_morphing_dnn_input_variables

    columns = create_DNN_columns_list(True, bkg_morphing_dnn_input_variables)
    for var in columns:
        print(var)
