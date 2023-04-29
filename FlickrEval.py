from LiMBeRModel import MultiModalModel
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image
import json

device = "cuda:0"


# ------------------------------ Building the Flickr8k Dataset ------------------------------ #
with open('data/Flickr8k_text/Flickr_8k.trainImages.txt', 'r') as f:
    train_image_names = f.read().split('\n')[:-1]

with open('data/Flickr8k_text/Flickr_8k.testImages.txt', 'r') as f:
    test_image_names = f.read().split('\n')[:-1]

with open('data/Flickr8k_text/Flickr8k.token.txt', 'r') as f:
    captions = f.read().split('\n')[:-1]

train_caption_dict = {
    image_name: []
    for image_name in train_image_names
}

test_caption_dict = {
    image_name: []
    for image_name in test_image_names
}
test_img_set = set(test_image_names)
train_img_set = set(train_image_names)


for caption in captions:
    image_name, caption = caption.split('#', maxsplit=1)
    cap_num = int(caption[0])
    caption = caption[2:]

    if image_name in train_img_set:
        train_caption_dict[image_name].append(caption)
    elif image_name in test_img_set:
        test_caption_dict[image_name].append(caption)

print("Done building Flickr8k dataset")
# Now we have test_caption_dict and train_caption_dict, which map image names to lists of captions
# Let's generate some captions for the test set and see how they do

model = MultiModalModel()
print("Done building model")

data = {}
i = 0
for image_name in test_image_names:
    img = Image.open('data/Flicker8k_Dataset/' + image_name)
    img = model.preprocess(img).to(device)

    my_cap = model.generate(img, decode=True)[0]
    my_cap = my_cap.lower().split()
    real_captions = [caption.lower().split() for caption in test_caption_dict[image_name]]
    bleu = sentence_bleu(real_captions, my_cap, weights=(0.95, 0.4, 0.1, 0))
    data[image_name] = {'bleu':bleu, 'my_cap':my_cap[:len(real_captions)], 'captions':real_captions}
    i += 1
    if i % 100 == 0:
        print(f"Done with {i} images out of {len(test_image_names)}")
        json.dump(data, open('Limber_flickr8k.json', 'w'))
json.dump(data, open('Limber_flickr8k.json', 'w'))

