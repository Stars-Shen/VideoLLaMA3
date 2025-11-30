import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# NOTE: transformers==4.46.3 is recommended for this script
# 这里既可以是 "DAMO-NLP-SG/VideoLLaMA3-7B"
# 也可以换成你本地的模型路径，比如 "/home/wqshen/models/VideoLLaMA3-7B"
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"

# =========================
# 1. 多卡 / 显存配置
# =========================

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    # 为每块可见 GPU 设置一个最大显存配额，防止顶满（按自己机器情况调整）
    max_memory = {i: "20GiB" for i in range(num_gpus)}
    # 可选：如果内存比较大，可以允许一部分权重 offload 到 CPU
    max_memory["cpu"] = "64GiB"
else:
    num_gpus = 0
    max_memory = None

# =========================
# 2. 加载模型（自动切到多卡）
# =========================

if num_gpus > 0:
    device_map = "auto"  # 让 accelerate 自动把模型拆到多卡
else:
    device_map = {"": "cpu"}  # 没有 GPU 就用 CPU（很慢，仅调试用）

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,          # bfloat16 节省显存
    attn_implementation="flash_attention_2",
    device_map=device_map,
    max_memory=max_memory,               # 控制每块卡显存上限（仅多卡时有用）
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 找到模型第一个 CUDA 参数所在的 device，后面把输入统一搬到这上面即可
if num_gpus > 0:
    first_device = next(p.device for p in model.parameters() if p.device.type == "cuda")
else:
    first_device = torch.device("cpu")


# =========================
# 3. 推理函数
# =========================

@torch.inference_mode()
def infer(conversation):
    # conversation: list[dict]，按官方示例格式
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # 把所有 tensor 输入搬到 first_device（通常是逻辑 cuda:0），
    # 模型其它部分由 accelerate 在多卡之间自动调度
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(first_device, non_blocking=True)

    # 图像 / 视频像素转成 bfloat16
    if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    # max_new_tokens 可以根据显存适当调小，比如 512、256
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
    )
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response


# =========================
# 4. 一些示例对话
# =========================

if __name__ == "__main__":
    # Video conversation
    video_conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": "../assets/video.mp4",
                        "fps": 1,
                        "max_frames": 180,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        # "What is the cat doing? Please describe the scene, "
                        # "the objects and the actions in detail."
                        # "Please describe the video in detail."
                        "Please help me summarize the content of the video in bullet points"
                    ),
                },
            ],
        },
    ]
    print("=== Video conversation ===")
    print(infer(video_conversation))
    print()

    # Image conversation
    image_conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": {"image_path": "../assets/sora.png"}},
                {"type": "text", "text": "Please describe the model?"},
            ],
        }
    ]
    print("=== Image conversation ===")
    print(infer(image_conversation))
    print()

    # Mixed conversation
    mixed_conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": "../assets/video.mp4",
                        "fps": 1,
                        "max_frames": 180,
                    },
                },
                {
                    "type": "text",
                    "text": "What is the relationship between the video and the following image?",
                },
                {"type": "image", "image": {"image_path": "../assets/sora.png"}},
            ],
        }
    ]
    print("=== Mixed conversation ===")
    print(infer(mixed_conversation))
    print()

    # Plain text conversation
    text_conversation = [
        {
            "role": "user",
            "content": "What is the color of bananas?",
        }
    ]
    print("=== Text conversation ===")
    print(infer(text_conversation))
