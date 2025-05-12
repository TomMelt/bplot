import click
import struct
import sys
import re
import numpy as np
import plotext as plt


DTYPE_SIZES = {
    "i": 4,
    "I": 4,
    "q": 8,
    "Q": 8,
    "f": 4,
    "d": 8,
}

DTYPE_EXPLANATION = {
    "i": "int",
    "I": "unsigned int",
    "q": "long long",
    "Q": "unsigned long long",
    "f": "float",
    "d": "double",
}

help_text = (
    ", ".join(
        [
            f"{dtype} = {DTYPE_EXPLANATION[dtype]} ({DTYPE_SIZES[dtype]})"
            for dtype in DTYPE_SIZES.keys()
        ]
    )
    + "."
)


@click.command()
@click.option(
    "--filename",
    "-f",
    type=click.File("r"),
    help="path to file containing binary data.",
    required=True,
)
@click.option(
    "--dtype",
    "-d",
    type=click.Choice(DTYPE_SIZES.keys()),
    default="d",
    help=help_text,
    required=True,
)
@click.option(
    "--shape",
    "-s",
    help="shape of data in format dim1,dim2,...dimN e.g., for a 2D array that 4 x 8 elements use --shape=4,8.",
    required=True,
)
@click.option(
    "--plot-range",
    "-r",
    default="",
    help="range of indices to plot e.g., for a 2D array to plot the first 8 elements from the first row use --plot-range=0,1:8.",
    required=True,
)
@click.option(
    "--layout",
    "-l",
    type=click.Choice(["f", "c"]),
    help="layout in memory, f = fortran or c = c/c++.",
    required=True,
)
@click.option(
    "--plot-height",
    type=int,
    default=0,
    help="set plot height. (default = 0, which will use full height of terminal).",
)
def main(filename, dtype, shape, plot_range, layout, plot_height):
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
    shape = np.array(list(map(int, re.findall(r"(\d+)", shape))))

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

    bin_data = read_binary_data(
        filename=filename.name, shape=shape, dtype=dtype, layout=layout
    )

    plot_data(data=bin_data[slice_], plot_height=plot_height)


def read_binary_data(filename, shape, dtype, layout):

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

    if layout == "f":
        bin_data = np.array(unpacked_data).reshape(shape[::-1]).transpose()
    else:
        bin_data = np.array(unpacked_data).reshape(shape)

    return bin_data


def plot_data(data, plot_height):

    (w, h) = plt.ts()  # find terminal width and height
    if plot_height > 0:
        h = plot_height
    plt.plot_size(w, h)
    plt.scatter(data)
    plt.title(" ".join(sys.argv))
    plt.show()


if __name__ == "__main__":
    main()
