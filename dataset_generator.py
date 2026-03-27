import argparse
import random


def generate_dataset(n, k, T, output_file="input.txt", seed=None):
    if seed is not None:
        random.seed(seed)

    assert 1 <= n <= 100000, "n out of bounds"
    assert 1 <= k <= 128, "k out of bounds"
    assert 1 <= T <= 50, "T out of bounds"

    used_points = set()

    with open(output_file, "w") as f:
        f.write(f"{n}\n")
        f.write(f"{k}\n")
        f.write(f"{T}\n")

        while len(used_points) < n:
            x = random.randint(-10000, 10000)
            y = random.randint(-10000, 10000)
            z = random.randint(-10000, 10000)

            if (x, y, z) in used_points:
                continue

            used_points.add((x, y, z))

            I = random.randint(0, 255)
            f.write(f"{x} {y} {z} {I}\n")

    print(f"Dataset written to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a point-cloud dataset for the assignment.")
    parser.add_argument("--n", "-n", type=int, default=10, help="Number of points.")
    parser.add_argument("--k", "-k", type=int, default=1, help="Number of neighbors/clusters.")
    parser.add_argument("--T", "-T", type=int, default=10, help="Number of K-means iterations.")
    parser.add_argument(
        "--output", "-o", default="input.txt", help="Path to the generated input file.")
    parser.add_argument("--seed", type=int, help="Optional RNG seed for reproducible datasets.")
    return parser.parse_args()


def main():
    args = parse_args()
    generate_dataset(args.n, args.k, args.T, output_file=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
