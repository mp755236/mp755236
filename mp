from diffusers import DiffusionPipeline
import argparse
import torch
import time
import bitsandbytes as bnb
import json

SHORT_NAME_MAPPER = {
    "stabilityai/stable-diffusion-3-medium-diffusers": "sd3",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": "pixart"
}


def load_pipeline(args):
    pipeline = DiffusionPipeline.from_pretrained(args.ckpt_id, torch_dtype=torch.float16).to("cuda")

    def replace_regular_linears(module, mode="8bit"):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                in_features = child.in_features
                out_features = child.out_features
                device = child.weight.data.device

                # Create and configure the Linear layer
                has_bias = True if child.bias is not None else False
                if mode == "8bit":
                    new_layer = bnb.nn.Linear8bitLt(in_features, out_features, bias=has_bias, has_fp16_weights=False)
                else:
                    # TODO: Make that configurable
                    # fp16 for compute dtype leads to faster inference
                    # and one should almost always use nf4 as a rule of thumb
                    bnb_4bit_compute_dtype = torch.float16
                    quant_type = "nf4"

                    new_layer = bnb.nn.Linear4bit(
                        in_features,
                        out_features,
                        bias=has_bias,
                        compute_dtype=bnb_4bit_compute_dtype,
                        quant_type=quant_type,
                    )

                new_layer.load_state_dict(child.state_dict())
                new_layer = new_layer.to(device)

                # Set the attribute
                setattr(module, name, new_layer)
            else:
                # Recursively apply to child modules
                replace_regular_linears(child, mode=mode)

    if args.mode is not None:
        replace_regular_linears(pipeline.transformer, args.mode)

    pipeline.set_progress_bar_config(disable=True)
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_id", default="stabilityai/stable-diffusion-3-medium-diffusers", type=str, choices=list(SHORT_NAME_MAPPER.keys()))
    parser.add_argument("--mode", default=None, type=str, choices=["8bit", "4bit"])
    parser.add_argument("--prompt", default="a golden vase with different flowers", type=str)
    args = parser.parse_args()

    torch.cuda.reset_peak_memory_stats()
    pipeline = load_pipeline(args)

    for _ in range(5):
        _ = pipeline(args.prompt, generator=torch.manual_seed(2024))

    start = time.time()
    output = pipeline(args.prompt, generator=torch.manual_seed(2024))
    end = time.time()
    mem_bytes = torch.cuda.max_memory_allocated()

    image = output.images[0]
    image_name = f"{SHORT_NAME_MAPPER[args.ckpt_id]}" + "_".join(args.prompt.split(" ")) 
    if args.mode is not None:
        image_name += f"_{args.mode}"
    image.save(f"{image_name}.png")

    print(f"Memory: {mem_bytes/(10**6):.3f} MB")
    print(f"Execution time: {(end - start):.3f} sec")

    info = dict(memory=f"{mem_bytes/(10**6):.3f}", time=f"{(end - start):.3f}")
    info_file = f"{SHORT_NAME_MAPPER[args.ckpt_id]}_info.json" if args.mode is None else f"{SHORT_NAME_MAPPER[args.ckpt_id]}_info_{args.mode}.json"
    with open(info_file, "w") as f:
        json.dump(info, f)
