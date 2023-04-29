
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from transformers import GPTJForCausalLM, AutoTokenizer
from PIL import Image
import clip
from typing import Callable
from functools import partial
from einops import rearrange

def top_k_filter(logits, k):
    assert k > 0
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs

def remove_tokens_after_eos(tensor, eos_token):
    # any tokens after and end of sequence token is produced are also set to the eos token, and removed
    eos_index = (tensor == eos_token).nonzero()
    if eos_index.any():
        tensor[eos_index[0] :] = eos_token

    tensor = tensor.tolist()
    return [i for i in tensor if (not i == eos_token)]

def top_p_filter(logits, threshold = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - threshold)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


class Lambda(torch.nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        assert hasattr(fn, "__call__")
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

class MultiModalModel(torch.nn.Module):
    def __init__(self, device="cuda:0"):
        super().__init__()
        self.device = device
        self.gptj = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", low_cpu_mem_usage=True).half().to(device)

        proj_dict = torch.load("projection.pt", map_location='cpu')

        self.projection = nn.Linear(3072, 4096)
        self.projection.load_state_dict(proj_dict)
        self.projection.half().to(device)

        


        self.clip = clip.load("RN50x16", device=device)[0].visual
        self.clip.attnpool = Lambda(
            partial(rearrange, pattern="b d h w -> b (h w) d")
        )  # remove attn pooling, just use reshaped features

        self.wte = self.gptj.transformer.wte
        
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id # = 50256
        self.tokenizer.padding_side = "right"

        def maybe_add_batch_dim(t):
            if t.ndim == 3:
                return t.unsqueeze(0)
            else:
                return t
            
        self.preprocess = T.Compose(
            [
                T.Resize(384, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(384),
                lambda image: image.convert("RGB"),
                T.ToTensor(),
                maybe_add_batch_dim,
                T.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
                lambda x: x.to(self.device),
            ]
        )


    @torch.no_grad()
    def forward(self, img, text = "an image of"):
        img = self.clip(img)
        img_projection = self.projection(img)
        text = self.tokenizer(text, return_tensors="pt").input_ids.to("cuda:0")
        text = self.wte(text)
        full_input = torch.cat([img_projection, text], dim=1)
        logits = self.gptj.forward(inputs_embeds=full_input, use_cache=True)
        return logits
    
    @torch.no_grad()
    def generate(
        self, img, text = "an image of", max_steps = 100, temperature = 1.0, top_k = 0, top_p = 0.9, eos_token = 50256, decode = True):
        clip_embeddings = self.clip(img)
        img_projection = self.projection(clip_embeddings)
        text = self.tokenizer(text, return_tensors="pt").input_ids.to("cuda:0")
        text = self.wte(text)
        embeddings = torch.cat([img_projection, text], dim=1)
        bs, s, _ = embeddings.shape
        # init output with image tokens
        out = torch.zeros((bs, s), dtype=torch.long).to(self.device) 
        # do sampling
        for i in range(max_steps):
            if i == 0:
                # initial input
                outputs = self.gptj.forward(
                    inputs_embeds=embeddings,
                    use_cache=True,
                )

            else:
                # now caching past k/v so we can use only the last token
                outputs = self.gptj.forward(
                    input_ids=out[:, -1:], use_cache=True, past_key_values=past_key_values, output_hidden_states = True
                )

            logits = outputs.logits[:, -1, :].float()#.cuda()
            past_key_values = outputs.past_key_values

            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1)
            else:
                if top_k > 0:
                    logits = top_k_filter(logits, k=top_k)
                if top_p > 0:
                    logits = top_p_filter(logits, threshold=top_p)
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat((out, next_token), dim=-1)
            if eos_token is not None and (next_token == eos_token).all():
                break
        out = out[:,s:]
        if decode:
            captions = []
            for b in out:
                b = remove_tokens_after_eos(b, eos_token)
                caption = self.tokenizer.decode(b)
                captions.append(caption)
            out = captions
        return out
    
@torch.no_grad()
def generate(
    self, img, text = "an image of", max_steps = 100, temperature = 1.0, top_k = 0, top_p = 0.9, eos_token = 50256, decode = True):
    clip_embeddings = self.clip(img)
    img_projection = self.projection(clip_embeddings)
    text = self.tokenizer(text, return_tensors="pt").input_ids.to("cuda:0")
    text = self.wte(text)
    embeddings = torch.cat([img_projection, text], dim=1)
    bs, s, _ = embeddings.shape
    # init output with image tokens
    out = torch.zeros((bs, s), dtype=torch.long).to(self.device) 
    # do sampling
    for i in range(max_steps):
        if i == 0:
            # initial input
            outputs = self.gptj.forward(
                inputs_embeds=embeddings,
                use_cache=True,
            )
            # all_states= outputs.hidden_states
            # l = layer_decode(model, all_states)
            # parse = [model.lm_head(state) for state in all_states]
            # # parse = [F.softmax(s[:, -1, :].float() / temperature, dim = -1) for s in parse]
            # # all_layers = torch.stack(parse)
            # # print("all_layers",all_layers.size())
            # # first_probs =  F.softmax(outputs.logits[:, -1, :].float() / temperature, dim = -1)
            # l = torch.stack(l)
            # l = F.softmax(l, dim = 1)
        else:
            # now caching past k/v so we can use only the last token
            outputs = self.gptj.forward(
                input_ids=out[:, -1:], use_cache=True, past_key_values=past_key_values, output_hidden_states = True
            )

        logits = outputs.logits[:, -1, :].float()#.cuda()
        past_key_values = outputs.past_key_values

        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1)
        else:
            if top_k > 0:
                logits = top_k_filter(logits, k=top_k)
            if top_p > 0:
                logits = top_p_filter(logits, threshold=top_p)
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        out = torch.cat((out, next_token), dim=-1)
        if eos_token is not None and (next_token == eos_token).all():
            break
    out = out[:,s:]
    if decode:
        captions = []
        for b in out:
            b = remove_tokens_after_eos(b, eos_token)
            caption = self.tokenizer.decode(b)
            captions.append(caption)
        out = captions
    return out
            
        

def get_flickr_data():
    image_path = 'data/Flicker8k_Dataset/'

    with open('data/Flickr8k_text/Flickr_8k.trainImages.txt', 'r') as f:
        train_image_names = f.read().split('\n')[:-1]

    with open('data/Flickr8k_text/Flickr_8k.testImages.txt', 'r') as f:
        test_image_names = f.read().split('\n')[:-1]

def im():
    img = "/gpfs/data/epavlick/datasets/mscoco/val2017/000000000285.jpg"
    img = Image.open(img)
    return img


if __name__ == "__main__":
    model = MultiModalModel(verbose=True)
    
    img = im()
    img = model.preprocessor(img)
    print(model.forward(img).logits.shape)
    print(model.generate(img))