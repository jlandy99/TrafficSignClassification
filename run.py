from preprocess import preprocess, rebalance
from Net import Net, printModel
from TrafficSignDataset import TrafficSignDataset, dataLoader
from train import train_model
from plot import plot


# Use this function if using GPU to run code
def setupGPU():
    """Make sure we are leveraging the hosted GPU"""
    if torch.cuda.is_available():
        print("Using the GPU. You are good to go!")
        device = torch.device('cpu')
    else:
        raise Exception("WARNING: Could not find GPU! Using CPU only.")
        device = torch.device('cuda:0')
        

def main():
    # Process our dataset
    X_processed, Y_processed = preprocess()
    # Normalize class distribution
    X_processed, Y_processed, TOTAL_IMAGES = rebalance(X_processed, Y_processed)
    # Print a summary of Neural Network
    net = printModel()
    # Create a dataset class and a data loader object to split into train, val, test
    train_loader, val_loader, test_loader = dataLoader(X_processed, Y_processed, TOTAL_IMAGES)
    # Train our model
    train_model(net, train_loader, val_loader)
    # Plot final results (accuracy and loss across epochs)
    plot(test_loader)


if __name__ == '__main__':
    main()
