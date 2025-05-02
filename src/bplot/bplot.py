import click
import struct
import sys
import re
import numpy as np
import plotext as plt


DTYPE_SIZES = {
    "f": 4,
    "d": 8,
}


@click.command()
@click.option(
    "--filename", "-f", help="path to file containing binary data.", required=True
)
@click.option(
    "--dtype", "-d", default="d", help="d = double (8), f = float (4).", required=True
)
@click.option(
    "--shape",
    "-s",
    help="shape of data in format dim1, dim2, ...dimN e.g., 4, 8.",
    required=True,
)
@click.option(
    "--plot-range",
    "-r",
    default="",
    help="range of indices to plot e.g., 1, 1:8.",
    required=True,
)
def main(filename, dtype, shape, plot_range):
    """
    \b
     █████               ████            █████
    ░░███               ░░███           ░░███
     ░███████  ████████  ░███   ██████  ███████
     ░███░░███░░███░░███ ░███  ███░░███░░░███░
     ░███ ░███ ░███ ░███ ░███ ░███ ░███  ░███
     ░███ ░███ ░███ ░███ ░███ ░███ ░███  ░███ ███
     ████████  ░███████  █████░░██████   ░░█████
    ░░░░░░░░   ░███░░░  ░░░░░  ░░░░░░     ░░░░░
               ░███
               █████
              ░░░░░
    \b
    bplot can be used to plot binary data ouput from debuggers like gdb, mdb etc.
    """
    shape = np.array(list(map(int, re.findall(r"(\d+)", shape)))).transpose()

    if len(plot_range) > 15:
        print("Error :: --plot-range should be less than 15 chars long")
        exit(1)
    elif not all(c in "1234567890-[],: " for c in plot_range):
        print(
            'Error :: --plot-range can only consist of the following chars: "1234567890-,: "'
        )
        exit(1)
    else:
        try:
            slice_ = eval(f"np.s_[{plot_range}]")
        except:
            print(f"Error :: cannot parse --plot-range [{plot_range!r}].")
            exit(1)

    # Open the file in binary read mode
    with open(filename, "rb") as file:
        bin_data = file.read()

    try:
        dtype_size = DTYPE_SIZES[dtype]
    except KeyError:
        print(
            f"Error :: dtype [{dtype}] not recognized. Supported types are {DTYPE_SIZES.keys()}."
        )

    # Ensure the data length is a multiple of dtype_size
    if len(bin_data) % dtype_size != 0:
        print(
            f"Warning :: file size ({len(bin_data)} bytes) is not a multiple of {dtype_size}. Truncating extra bytes."
        )
        bin_data = bin_data[: len(bin_data) - (len(bin_data) % dtype_size)]

    # Calculate number of doubles
    num_values = len(bin_data) // dtype_size

    # Unpack all doubles
    unpacked_data = struct.unpack(f"{num_values}{dtype}", bin_data)

    bin_data = np.array(unpacked_data).reshape(shape.transpose()).transpose()
    plt.scatter(bin_data[slice_])
    plt.title(" ".join(sys.argv))
    plt.show()


if __name__ == "__main__":
    main()
