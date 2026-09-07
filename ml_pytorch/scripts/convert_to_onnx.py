import spox.opset.ai.onnx.v17 as op
from spox import argument, build, inline, Tensor
import os
import sys
import onnx
import numpy as np
import argparse
import onnxruntime as ort
import uproot
import importlib

parser = argparse.ArgumentParser(description="Convert keras to onnx or average models")
parser.add_argument("-i", "--input", type=str, required=True, help="Input directory")
parser.add_argument("-o", "--output", type=str, default=None, help="Output directory")
parser.add_argument(
    "-ar",
    "--average-ratio",
    action="store_true",
    default=False,
    help="Perform the average between the models in the directory of the ratios of the outputs",
)
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    default=False,
    help="Perform the average between two models and compare the output",
)
parser.add_argument(
    "-mt",
    "--model_type",
    default="onnx",
    help="Parameter to determine, what type of model is being converted (onnx or keras)",
)
parser.add_argument(
    "-v",
    "--input-variables",
    default="bkg_morphing_dnn_input_variables",
    help="Input variables module name (e.g. bkg_morphing_dnn_input_variables). Ignored if --config is provided and contains input_variables.",
)
parser.add_argument(
    "-c",
    "--config",
    default=None,
    help="Training config YAML file. If provided, reads input_variables from it (takes precedence over --input-variables).",
)
parser.add_argument(
    "-mc",
    "--multiclass",
    action="store_true",
    default=False,
    help=(
        "Treat the model as a 4-node QCD-morphing classifier with output nodes "
        "(data_num, data_den, ttbar_num, ttbar_den) in class_idx order. The "
        "weight is w = relu(r_0 - alpha * r_2) / relu(r_1 - alpha_den * r_3): "
        "the ttbar-subtracted data shape in the numerator region over the "
        "ttbar-subtracted data shape in the denominator region. Applied to raw "
        "data in the denominator region it reproduces the QCD shape of the "
        "numerator region."
    ),
)
parser.add_argument(
    "-al",
    "--alpha",
    type=float,
    default=0.0,
    help=(
        "Multiclass only. ttbar-to-data yield ratio (N_ttbar / N_data, a single "
        "number from the cutflow) in the numerator region. Scales the ttbar node "
        "r_2 before it is subtracted from the data node r_0, which is required "
        "because training balances every class to equal total weight. 0.0 "
        "(default) disables the numerator subtraction."
    ),
)
parser.add_argument(
    "-ald",
    "--alpha-den",
    type=float,
    default=0.0,
    help=(
        "Multiclass only. Same as --alpha but for the denominator region: the "
        "ttbar-to-data yield ratio (N_ttbar / N_data) in the denominator region "
        "(boosted_control_sideband_region_D). Scales the ttbar node r_3 before "
        "it is subtracted from the data node r_1, giving "
        "w = relu(r_0 - alpha * r_2) / relu(r_1 - alpha_den * r_3). 0.0 "
        "(default) disables the denominator subtraction and gives w = "
        "relu(r_0 - alpha * r_2) / r_1."
    ),
)
args = parser.parse_args()

SAVE_SINGLE_RATIOS = False

# Numerical guards for the multiclass morphing-weight ratio (see
# get_multiclass_ratio_model_tensor_onnx). The denominator is a relu, so it is
# exactly 0 for every event the classifier considers more ttbar- than QCD-like
# in the denominator region; relu(num) / 0 is +inf (or 0/0 -> nan). Because
# main() averages the per-model weights *after* the division, a single sub-model
# hitting 0 poisons the ensemble weight for that event. RATIO_DEN_EPS floors the
# denominator so no inf/nan is produced; RATIO_W_MAX bounds the resulting
# per-model weight so a near-zero denominator gives a large-but-finite value
# instead of a spike that dominates avg_w and any downstream histogram bin.
# Lower RATIO_W_MAX if a few high-weight events still dominate downstream.
RATIO_DEN_EPS = 1e-6

if args.model_type == "keras":
    import tensorflow as tf
    import tf2onnx

input_variables_name = args.input_variables
if args.config is not None:
    from omegaconf import OmegaConf

    _cfg = OmegaConf.load(args.config)
    if _cfg.get("input_variables") is not None:
        input_variables_name = _cfg.input_variables

dnn_input_variables_module = importlib.import_module(
    f"ml_pytorch.defaults.{input_variables_name}"
)
dnn_input_variables = dnn_input_variables_module.dnn_input_variables
print(f"Input variables: {dnn_input_variables}")

columns = list(dnn_input_variables.keys())


def save_onnx_model(onnx_model_final, onnx_model_name):
    if os.path.exists(onnx_model_name):
        print(f"Removing {onnx_model_name}")
        os.remove(onnx_model_name)
    onnx.save(onnx_model_final, onnx_model_name)
    print(f"Model saved as {onnx_model_name}")


def get_onnx_output(onnx_model_name, input_data):
    sess_options = ort.SessionOptions()

    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1

    session = ort.InferenceSession(
        onnx_model_name, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    # print the input/output name and shape
    input_name = [input.name for input in session.get_inputs()]
    output_name = [output.name for output in session.get_outputs()]

    input_shape = [input.shape for input in session.get_inputs()]
    output_shape = [output.shape for output in session.get_outputs()]

    input_example = {input_name[0]: input_data}
    output_onnx = session.run(output_name, input_example)

    return output_onnx


def load_events():
    # load a root file
    file_name = "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/file_root/JetMET_2022EE_2b_signal_region_to_4b_soumya_january2025.root"
    tree = uproot.open(file_name)["tree"]
    input_data_dict = tree.arrays(columns, library="np")
    n_events = 10
    # get the input data as a numpy array
    input_data = np.array(
        [input_data_dict[col][:n_events] for col in columns], dtype=np.float32
    ).T

    return input_data


def compare_output_onnx_keras(onnx_model_name, keras_model):
    input_data = load_events()

    output_onnx = get_onnx_output(onnx_model_name, input_data)[0]

    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    # input_tensor = tf.expand_dims(input_tensor, 0)
    output_keras = keras_model.predict(input_tensor)

    print(output_onnx)
    print(output_keras)

    assert np.allclose(output_onnx, output_keras, rtol=1e-03, atol=1e-05)


def compare_output_onnx_ratio(
    onnx_model_name, onnx_model_ratio_name, onnx_model_name_2, average
):
    input_data = load_events()
    print("#############################\n\n compare output")

    output_onnx = get_onnx_output(onnx_model_name, input_data)[0]
    output_onnx_ratio = output_onnx[:, 1] / output_onnx[:, 0]
    print("output_onnx", output_onnx)
    print("output_onnx by hand ratio", output_onnx[:, 1] / output_onnx[:, 0])

    if onnx_model_name_2:
        output_onnx_2 = get_onnx_output(onnx_model_name_2, input_data)[0]
        output_onnx_ratio_2 = output_onnx_2[:, 1] / output_onnx_2[:, 0]
        print("output_onnx_2", output_onnx_2)
        print("output_onnx_2 by hand ratio", output_onnx_2[:, 1] / output_onnx_2[:, 0])
        if average:
            output_by_hand = (output_onnx_ratio + output_onnx_ratio_2) / 2
        else:
            output_by_hand = [output_onnx_ratio, output_onnx_ratio_2]
        print("\noutput_by_hand by hand", output_by_hand)

    output_onnx_ratio = get_onnx_output(onnx_model_ratio_name, input_data)
    print("output onnx directly from pad", output_onnx_ratio)
    print("for ", onnx_model_ratio_name)

    assert np.allclose(
        output_onnx_ratio, output_by_hand if onnx_model_name_2 else output_onnx_ratio
    )


def get_ratio_model_tensor_onnx(onnx_model, b):
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)

    # get the output shape of the model
    output_shape = (
        inferred_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
    )
    print(f"Output shape: {output_shape}")

    # To take the ratio of the first model too.
    (r,) = inline(onnx_model)(b).values()
    print(f"{b.type = !s}, {r.type = !s}")
    if output_shape == 1:
        r = op.div(r, op.sub(op.const(1.0, dtype="float32"), r))
    elif output_shape == 2:
        r_0 = op.squeeze(
            op.slice(
                r,
                op.constant(value=np.array([0, 0])),
                op.constant(value=np.array([sys.maxsize, 1])),
            ),
            axes=op.const([-1]),
        )
        r_1 = op.squeeze(
            op.slice(
                r,
                op.constant(value=np.array([0, 1])),
                op.constant(value=np.array([sys.maxsize, 2])),
            ),
            axes=op.const([-1]),
        )
        r = op.div(r_1, r_0)
        # r = op.div(r_0, r_1)
        print(f"{r_0.type = !s}, {r_1.type = !s}, {r.type = !s}")
    else:
        raise ValueError("The output shape is not 1 or 2")

    return r

def get_multiclass_ratio_model_tensor_onnx(onnx_model, b, alpha, alpha_den):
    """Build the QCD-morphing weight tensor for a 4-node multiclass model.

    Output-node convention (class_idx order, i.e. sorted by ``lbl``)::

        r_0 = p(data,  numerator region)    e.g. boosted_control_region_C
        r_1 = p(data,  denominator region)  e.g. boosted_control_sideband_region_D
        r_2 = p(ttbar, numerator region)
        r_3 = p(ttbar, denominator region)

    The weight reweights data in the denominator region so that it reproduces
    the QCD shape of the numerator region::

        w(x) = clip(relu(r_0 - alpha * r_2) / (relu(r_1 - alpha_den * r_3) + eps),
                    0, RATIO_W_MAX)

    ``alpha = N_ttbar / N_data`` in the numerator region and ``alpha_den =
    N_ttbar / N_data`` in the denominator region. They are needed because
    training balances every class to equal total weight, so ``r_0``..``r_3``
    are all unit-normalised shapes; the alphas restore the physical ttbar
    fraction before each subtraction. ``alpha = 0`` / ``alpha_den = 0`` disable
    the corresponding subtraction (``alpha = alpha_den = 0`` gives a pure
    data-shape morph ``w = r_0 / r_1``). Only the shape of ``w`` matters; its
    overall scale is fixed downstream when the morphed sample is renormalised.

    ``alpha_den`` must match the density of the sample the weight is multiplied
    onto downstream: use ``alpha_den = 0`` when reweighting *raw* data in the
    denominator region (denominator = data-D density ``r_1``), and
    ``alpha_den = N_ttbar/N_data`` in region D when reweighting a
    ttbar-subtracted ``data_D - ttbar_D_MC`` sample (denominator = QCD-D
    density).

    ``eps`` (``RATIO_DEN_EPS``) floors the denominator: ``relu(r_1 - alpha_den *
    r_3)`` is exactly 0 for every event the classifier deems more ttbar- than
    QCD-like in the denominator region, and ``relu(num) / 0`` is ``+inf`` (or
    ``0/0 -> nan``). Since ``main()`` averages the per-model weights *after* this
    division, one sub-model hitting 0 would poison the ensemble weight for that
    event. The final ``clip`` to ``RATIO_W_MAX`` keeps a near-zero denominator
    from turning into a spike that dominates ``avg_w`` and any downstream bin.
    """
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)

    # get the output shape of the model
    output_shape = (
        inferred_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
    )
    print(f"Output shape: {output_shape}")
    if output_shape != 4:
        raise ValueError(
            f"Multiclass morphing expects a 4-node model, got {output_shape} outputs"
        )

    # To take the ratio of the first model too.
    (r,) = inline(onnx_model)(b).values()
    print(f"{b.type = !s}, {r.type = !s}")

    def _node(i):
        return op.squeeze(
            op.slice(
                r,
                op.constant(value=np.array([0, i])),
                op.constant(value=np.array([sys.maxsize, i + 1])),
            ),
            axes=op.const([-1]),
        )

    r_0 = _node(0)  # data,  numerator region
    r_1 = _node(1)  # data,  denominator region
    r_2 = _node(2)  # ttbar, numerator region
    r_3 = _node(3)  # ttbar, denominator region

    if alpha:
        # relu guards against the subtraction overshooting where the classifier
        # thinks ttbar dominates the data shape (would give a negative weight)
        numerator = op.relu(
            op.sub(r_0, op.mul(op.const(float(alpha), dtype="float32"), r_2))
        )
    else:
        numerator = r_0
    if alpha_den:
        denominator = op.relu(
            op.sub(r_1, op.mul(op.const(float(alpha_den), dtype="float32"), r_3))
        )
    else:
        denominator = r_1
    # floor the denominator so a relu that clamps to 0 cannot produce inf/nan
    denominator = op.add(denominator, op.const(float(RATIO_DEN_EPS), dtype="float32"))
    r = op.div(numerator, denominator)
    # bound the per-model weight: a near-zero denominator would otherwise give a
    # finite-but-huge spike that dominates the averaged weight downstream
    print(
        f"{r_0.type = !s}, {r_1.type = !s}, {r_2.type = !s}, {r_3.type = !s}, "
        f"{r.type = !s}"
    )
    return r


def main():
    if args.input.endswith(".onnx") or args.input.endswith(".keras"):
        in_dir = os.path.dirname(args.input)
        model_files = [os.path.basename(args.input)]
        args.model_type = "keras" if args.input.endswith(".keras") else "onnx"
    else:
        in_dir = args.input

        model_files = [x for x in os.listdir(in_dir) if x.endswith(args.model_type)]
        model_files = [
            x
            for x in model_files
            if "average_model_from" not in x and "all_ratios" not in x
        ]

    out_dir = args.output if args.output else in_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.debug:
        model_files = model_files[:2]

    print(model_files)
    print("Lenght of input", len(columns))

    if args.average_ratio:
        print(f"Processing {model_files[0]}")

        tot_len = 1
        first_file_name = os.path.join(in_dir, model_files[0])
        b = argument(Tensor(np.float32, ("N", len(columns))))
        if args.model_type == "keras":
            model = tf.keras.models.load_model(first_file_name)
            model_ratio = tf.keras.models.Model(
                inputs=model.input, outputs=model.output[:, 1] / model.output[:, 0]
            )

            onnx_model_ratio_sum, _ = tf2onnx.convert.from_keras(
                model_ratio,
                input_signature=[
                    tf.TensorSpec(shape=(None, len(columns)), dtype=tf.float32)
                ],
            )

        elif args.model_type == "onnx":
            onnx_model_ratio_sum = onnx.load(first_file_name)

            if not args.multiclass:
                r = get_ratio_model_tensor_onnx(onnx_model_ratio_sum, b)
            else:
                r = get_multiclass_ratio_model_tensor_onnx(
                    onnx_model_ratio_sum, b, args.alpha, args.alpha_den
                )
            r_list = []
            r_list.append(r)

            onnx_model_ratio_sum = build({"args_0": b}, {"sum_w": r})
            onnx_model_ratio_list = onnx_model_ratio_sum

            if SAVE_SINGLE_RATIOS:
                save_onnx_model(
                    onnx_model_ratio_sum, f"{out_dir}/ratio_{model_files[0]}"
                )

        second_file_name = None
        if len(model_files) > 1:
            second_file_name = os.path.join(in_dir, model_files[1])
            for model_file in model_files[1:]:
                tot_len += 1
                print(f"\n\nAdding {model_file}")
                if args.model_type == "keras":
                    model_add = tf.keras.models.load_model(
                        os.path.join(in_dir, model_file)
                    )
                    model_ratio_add = tf.keras.models.Model(
                        inputs=model_add.input,
                        outputs=model_add.output[:, 0] / 1 - model_add.output[:, 1],
                    )

                    onnx_model_ratio_add, _ = tf2onnx.convert.from_keras(
                        model_ratio_add,
                        input_signature=[
                            tf.TensorSpec(shape=(None, len(columns)), dtype=tf.float32)
                        ],
                    )

                elif args.model_type == "onnx":
                    onnx_model_ratio_add = onnx.load(os.path.join(in_dir, model_file))

                print(b)
                (r,) = inline(onnx_model_ratio_sum)(b).values()
                if args.model_type == "keras":
                    (r1,) = inline(onnx_model_ratio_add)(b).values()
                if args.model_type == "onnx":
                    # r1 = op.div(r1, op.sub(op.const(1.0, dtype="float32"), r1))
                    if not args.multiclass:
                        r1 = get_ratio_model_tensor_onnx(onnx_model_ratio_add, b)
                    else:
                        r1 = get_multiclass_ratio_model_tensor_onnx(
                            onnx_model_ratio_add, b, args.alpha, args.alpha_den
                        )
                    r_list.append(r1)

                    if SAVE_SINGLE_RATIOS:
                        onnx_model_ratio = build({"args_0": b}, {"ratio_w": r1})
                        save_onnx_model(
                            onnx_model_ratio, f"{out_dir}/ratio_{model_file}"
                        )
                print(r)
                print(r1)

                s = op.add(r, r1)

                onnx_model_ratio_sum = build({"args_0": b}, {"sum_w": s})

        print(f"\ntotal length: {tot_len}")

        # Output all individual ratios
        onnx_model_ratio_list = build(
            {"args_0": b}, {f"ratio_{i}": r for i, r in enumerate(r_list)}
        )
        onnx_model_ratios_name = f"{out_dir}/all_ratios_model_{args.model_type}.onnx"
        save_onnx_model(onnx_model_ratio_list, onnx_model_ratios_name)

        (r_sum,) = inline(onnx_model_ratio_sum)(b).values()
        a = op.div(r_sum, op.constant(value_float=tot_len))

        onnx_model_final = build({"args_0": b}, {"avg_w": a})
        onnx_model_final_name = (
            f"{out_dir}/average_model_from_{args.model_type}.onnx"
            if not args.debug
            else f"{out_dir}/debug.onnx"
        )
        save_onnx_model(onnx_model_final, onnx_model_final_name)
        if args.model_type == "onnx" and args.debug:
            try:
                compare_output_onnx_ratio(
                    first_file_name, onnx_model_final_name, second_file_name, True
                )
                compare_output_onnx_ratio(
                    first_file_name, onnx_model_ratios_name, second_file_name, False
                )
            except uproot.exceptions.KeyInFileError:
                print(
                    "WARNING: The model is not compatible with the input data. Skipping comparison."
                )

        if args.debug:
            # rm the deubg.onnx model
            os.remove(onnx_model_final_name)

    else:
        for model_file in model_files:
            print(f"Processing {model_file}")
            model = tf.keras.models.load_model(os.path.join(in_dir, model_file))
            onnx_model_final_name = f"{out_dir}/{model_file.replace('.keras', '.onnx')}"

            input_signature = tf.TensorSpec(
                shape=(None, len(columns)), dtype=tf.float32
            )

            if "2.16" in tf.__version__:
                output_name = model.layers[-1].name

                @tf.function(input_signature=[input_signature])
                def _wrapped_model(input_data):
                    return {output_name: model(input_data)}

                onnx_model, _ = tf2onnx.convert.from_function(
                    _wrapped_model,
                    input_signature=[input_signature],
                )
            else:
                onnx_model, _ = tf2onnx.convert.from_keras(
                    model,
                    input_signature=[input_signature],
                )
            save_onnx_model(onnx_model, onnx_model_final_name)

            compare_output_onnx_keras(onnx_model_final_name, model)


if __name__ == "__main__":
    main()
