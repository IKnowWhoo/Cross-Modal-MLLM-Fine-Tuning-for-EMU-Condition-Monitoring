from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(

    "/data1/qwen/train", torch_dtype="auto", device_map="auto"  # 修改模型权重文件路径！！！！！！

)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(

#     "Qwen/Qwen2.5-VL-7B-Instruct",

#     torch_dtype=torch.bfloat16,

#     attn_implementation="flash_attention_2",

#     device_map="auto",

# )

# default processor

processor = AutoProcessor.from_pretrained("/data1/qwen/train")  # 修改模型权重文件路径！！！！！！

# The default range for the number of visual tokens per image in the model is 4-16384.    

# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.

# min_pixels = 256*28*28

# max_pixels = 1280*28*28

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [

    {

        "role": "user",

        "content": [

            {

                "type": "image",

                "image":"/data1/detectron2/data/1620102/bumingyeti/47359e39-393f-4c22-bcf7-677409b7d80e/ori014_6_GUANGZHOUNANZHANJINGGUANGXIAXING.jpg",
                #"/data1/qwen/funny_image.jpeg",  # 修改测试图片！！！！！！

            },
           # {
           #     "type": "image",
           #     "image":"/data1/detectron2/data/1620102/bumingyeti/83a36a4b-6bac-43e3-b059-a95a2e1caf70/ori005_7_CHONGQINGXIYUQIANSHANGXING.jpg",
           # },

            {"type": "text", "text": "Provide a detailed description of the image."},

        ],

    }

]

# Preparation for inference

text = processor.apply_chat_template(

    messages, tokenize=False, add_generation_prompt=True

)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(

    text=[text],

    images=image_inputs,

    videos=video_inputs,

    padding=True,

    return_tensors="pt",

)

inputs = inputs.to(model.device)

# Inference: Generation of the output

generated_ids = model.generate(**inputs, max_new_tokens=256)

generated_ids_trimmed = [

    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)

]

output_text = processor.batch_decode(

    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False

)

print(output_text)
