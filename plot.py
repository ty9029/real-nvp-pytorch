import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="RealNVP")
    parser.add_argument("--inference", type=str)
    parser.add_argument("--generate", type=str)
    args = parser.parse_args()

    fig, axis = plt.subplots(2, 2)

    data_i = np.load(args.inference)
    data_g = np.load(args.generate)

    axis[0, 0].scatter(data_i["x"][:, 0], data_i["x"][:, 1])
    xlim = axis[0, 0].get_xlim()
    ylim = axis[0, 0].get_ylim()
    axis[0, 0].set_title(r"$X \sim p(X)$")

    axis[1, 0].scatter(data_g["x"][:, 0], data_g["x"][:, 1])
    axis[1, 0].set_xlim(*xlim)
    axis[1, 0].set_ylim(*ylim)
    axis[1, 0].set_title(r"$X = g(z)$")

    axis[1, 1].scatter(data_g["z"][:, 0], data_g["z"][:, 1])
    xlim = axis[1, 1].get_xlim()
    ylim = axis[1, 1].get_ylim()
    axis[1, 1].set_title(r"$z \sim p(z)$")

    axis[0, 1].scatter(data_i["z"][:, 0], data_i["z"][:, 1])
    axis[0, 1].set_xlim(*xlim)
    axis[0, 1].set_ylim(*ylim)
    axis[0, 1].set_title(r"$z = f(X)$")

    fig.tight_layout()
    plt.savefig("results/output.jpg")


if __name__ == "__main__":
    main()
