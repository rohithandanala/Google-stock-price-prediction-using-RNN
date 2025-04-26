import matplotlib.pyplot as plt

def plot_predictions(real, predicted, save_path=None):
    plt.figure(figsize=(10,6))
    plt.plot(real, color='red', label='Real Stock Price')
    plt.plot(predicted, color='blue', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()  # Always close the plot to avoid memory leak
