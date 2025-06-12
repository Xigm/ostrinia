
import numpy as np

def plot_predictions_test(predictor, data_module, run_dir, log_metrics=None):
    """
    Plot predictions on the test set.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Create directory for plots
    plot_dir = Path(run_dir) / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Get test data
    test_data = data_module.test_dataloader()

    pred = []
    # Get predictions
    for i, batch in enumerate(test_data):
        batch = batch.to(predictor.device)
        predictions = predictor.predict_step(batch, i)
        pred.append(predictions)

    # stack list of tensors to a single tensor
    y, y_hat = [], []
    for element in pred:
        y.append(element['y'].cpu().numpy())
        y_hat.append(element['y_hat'].cpu().numpy())
    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)

    # take nodes which y[:, 0, n, :] sum is not zero
    non_zero_nodes = [i for i in range(y.shape[2]) if np.sum(y[:, 0, i, :]) != 0]

    # important nodes are the 3 ones with the highest sum across all time steps
    important_nodes = [4, 9, 10]

    # get indices of non-zero nodes
    horizon = y.shape[1]

    # get only one node
    y = y[:, :, :, 0]
    y_hat = y_hat[:, :, :, 0]

    if horizon > 6:
        horizons = range(0, horizon, 2)
    else:
        horizons = range(horizon)

    # Plot predictions
    for h in horizons:
        for n in non_zero_nodes:
            plt.figure(figsize=(10, 5))

            # create time indexes. the signal should spawn from the first non zero value to the first zero value after that
            # for the first index, find the first non-zero value
            start_index = np.where(y[:, h, n] != 0)[0][0]
            # for the last index, find the first zero value after the first non-zero value
            end_index = np.where(y[start_index:, h, n] == 0)[0]

            if end_index.size > 0:
                end_index = end_index[0] + start_index
            else:
                end_index = y.shape[0]

            if start_index != 0:
                start_index = start_index - 20

            plt.plot(range(start_index, end_index), y[start_index:end_index, h, n], label='True')
            plt.plot(range(start_index, end_index), y_hat[start_index:end_index, h, n], label='Predicted')
            plt.title(f' Node {n} - Horizon {h}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            if n in important_nodes:
                plt.title(f'Important Node {n} - Horizon {h}')
                plt.savefig(plot_dir / f'important_node_{n}_horizon_{h}.png')
            else:
                plt.title(f'Node {n} - Horizon {h}')
                plt.savefig(plot_dir / f'node_{n}_horizon_{h}.png')
            print(f'Saved plot for Node {n} - Horizon {h}')
            plt.close()

    print(f'Plots saved to {plot_dir}')