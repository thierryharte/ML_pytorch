import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c",
    "--config",
    required=False,
    default=None,
    help="Path to the configuration file",
    type=str,
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=None,
    help="Batch size",
    type=int,
)
parser.add_argument(
    "-e",
    "--epochs",
    default=None,
    help="Number of epochs",
    type=int,
)
parser.add_argument(
    "-n",
    "--num-workers",
    default=None,
    help="Number of workers for data loading",
    type=int,
)
parser.add_argument(
    "-d",
    "--data-dirs",
    nargs="+",
    default=None,
    help="Directory for data",
)
parser.add_argument(
    "-ev",
    "--eval",
    action="store_true",
    help="Evaluate the model",
    default=None,
)
parser.add_argument(
    "-sm",
    "--save-model",
    action="store_true",
    help="Save also the model next to the state dict",
    default=False,
)
parser.add_argument(
    "-ct",
    "--comet-token",
    required=False,
    default=None,
    help="Token to register job on comet for job monitoring (`comet.com`)",
    type=str,
)
parser.add_argument(
    "-cn",
    "--comet-name",
    required=False,
    default=None,
    help="Username to register job on comet for job monitoring (`comet.com`)",
    type=str,
)
parser.add_argument(
    "-cw",
    "--comet-workspace",
    required=False,
    default=None,
    help="Workspace in which to register the model",
    type=str,
)
parser.add_argument(
    "-ctg",
    "--comet-tags",
    required=False,
    nargs="+",
    default=None,
    help="Username to register job on comet for job monitoring (`comet.com`)",
    type=str,
)
parser.add_argument(
    "-g", "--gpus", default=None, help="GPU numbers separated by a comma", type=str
)
parser.add_argument(
    "--histos", default=None, help="Make histograms of sig and bkg output distribution", action="store_true"
)
parser.add_argument(
    "--roc", default=None, help="Make roc curve of discrimination", action="store_true"
)
parser.add_argument(
    "--history", default=None, help="Plot training history", action="store_true"
)
parser.add_argument(
    "-em",
    "--eval-model",
    default=None,
    help="Path to model to evaluate",
    type=str,
)
parser.add_argument(
    "-l",
    "--load-model",
    default=None,
    help="Path to model to load and continue training. The model should be the state at the best epoch",
    type=str,
)
parser.add_argument(
    "--onnx",
    default=None,
    help="Save model in ONNX format",
    action="store_true",
)
parser.add_argument(
    "--pin-memory",
    default=None,
    help="Pin memory for data loading",
    action="store_true",
)
parser.add_argument(
    "--overwrite",
    default=None,
    help="Overwrite output directory",
    action="store_true",
)
parser.add_argument(
    "-o",
    "--output-dir",
    default=None,
    help="Output directory",
    type=str,
)
parser.add_argument(
    "-s",
    "--seed",
    default=None,
    help="Seed for event shuffling and weights initialization",
    type=str,
)
parser.add_argument(
    "-s-n",
    "--save-numpy",
    default=None,
    help="Save numpy arrays of the output scores",
    action="store_true",
)
parser.add_argument(
    "--novars",
    action="store_true",
    help="If true, old save format without saved variations is expected",
    default=False,
)

# parser.print_help()
args = parser.parse_args()
