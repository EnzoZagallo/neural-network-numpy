import numpy as np

from data import generate_spiral
from layers import Dense
from activations import ReLU, Sigmoid, Softmax
from model import NeuralNetwork


def main():
    # ---- Configuration ----
    #Fix the random number generator for reproducibility 

    np.random.seed(42)

    SAMPLES_PER_CLASS = 200
    NUM_CLASSES = 3
    NOISE = 0.15
    EPOCHS = 2000
    LEARNING_RATE = 1.0
    PRINT_EVERY = 200

    # ---- Generate Data ----
    print("Generating spiral dataset...")
    X, y = generate_spiral(
        samples_per_class=SAMPLES_PER_CLASS,
        num_classes=NUM_CLASSES,
        noise=NOISE,
    )
    print(f"Dataset: {X.shape[0]} samples, {NUM_CLASSES} classes, "
          f"{X.shape[1]} features\n")

    # ---- Build Network ----
    # Architecture: 2 -> 64 (ReLU) -> 32 (ReLU) -> 3 (Softmax)
    model = NeuralNetwork()
    model.add(Dense(n_inputs=2, n_neurons=64), ReLU())
    model.add(Dense(n_inputs=64, n_neurons=32), ReLU())
    model.add(Dense(n_inputs=32, n_neurons=NUM_CLASSES), Softmax())

    print("Network architecture:")
    print("  Input  -> Dense(2, 64)  -> ReLU")
    print("  Hidden -> Dense(64, 32) -> ReLU")
    print("  Output -> Dense(32, 3)  -> Softmax")
    print(f"  Loss   -> Cross-Entropy")
    print(f"  Optimizer -> Stochastic Gradient Descent (lr={LEARNING_RATE})")
    print("-" * 50)

    # ---- Train ----
    model.train(X, y, epochs=EPOCHS, learning_rate=LEARNING_RATE,
                print_every=PRINT_EVERY)

    # ---- Visualize (optional — requires matplotlib) ----
    try:
        import matplotlib.pyplot as plt
#Create a figure with 3 subplots: dataset, decision boundary, and training curves
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1) Dataset, using the first and second columns of the dataset to respectively be the X and Y axis coordinates of the scatter plot
        axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="brg", s=10, alpha=0.7)
        axes[0].set_title("Spiral Dataset (3 classes)")
        axes[0].set_xlabel("x1")
        axes[0].set_ylabel("x2")

        # 2) Decision boundary, what would the network predict for possible point in the 2D Space?
        h = 0.01
        #Padding min and max for both x and y axis
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        #We create evenly spaced values along axis X and Y and they are used to create every possible combination of coordinate pairs. 
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h),
        )
        #We feed these coordinate pairs, in 2D grids,  so we convert using np.c_ and .raavel() to create a input in shape (N_samples, 2), a flat list of coordinate pairs
        grid_input = np.c_[xx.ravel(), yy.ravel()]
        #Storing every layers output
        model.forward(grid_input)
        #_ receives Softmax, while last_layer receives Dense 3
        last_layer, _ = model.layers[-1]
        #We need the last layer output to be the predicted probabilities for each class, as we dont need the loss function output, we can pass dummy labels
        _ = model.loss_fn.forward(last_layer.output, np.zeros(len(grid_input), dtype=int))
        # We get the fotmax proabilities for every grid point and voncerts baack to the 2D grid shape, so that contourf can draw the colored regions
        Z = np.argmax(model.loss_fn.output, axis=1).reshape(xx.shape)
        #We draw filled countour regions, colored according to the predicted class of Z, we then scatter the original dataset and set the labels and title

        axes[1].contourf(xx, yy, Z, cmap="brg", alpha=0.3)
        axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap="brg", s=10, alpha=0.7)
        axes[1].set_title("Decision Boundary")
        axes[1].set_xlabel("x1")
        axes[1].set_ylabel("x2")

        # 3) Training curves
        ax_loss = axes[2]
        ax_acc = ax_loss.twinx()
        epochs_range = range(1, len(model.training_history["loss"]) + 1)
        ax_loss.plot(epochs_range, model.training_history["loss"],
                     "r-", label="Loss")
        ax_acc.plot(epochs_range, model.training_history["accuracy"],
                    "b-", label="Accuracy")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss", color="r")
        ax_acc.set_ylabel("Accuracy", color="b")
        ax_loss.set_title("Training Progress")
        lines1, labels1 = ax_loss.get_legend_handles_labels()
        lines2, labels2 = ax_acc.get_legend_handles_labels()
        ax_loss.legend(lines1 + lines2, labels1 + labels2, loc="center right")

        plt.tight_layout()
        plt.savefig("images/training-results.png", dpi=150)
        print("\nPlot saved to images/training-results.png")
        plt.show()

    except ImportError:
        print("\nInstall matplotlib to see visualizations: pip install matplotlib")


if __name__ == "__main__":
    main()
