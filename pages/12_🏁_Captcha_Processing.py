import streamlit as st
from collections import OrderedDict
from PIL import Image
import torch, torchvision
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Xử lý Captcha", page_icon="🏁")

st.header("Xử lý Captcha")

decoding_dict = {
    0: 'a',
    1: 'f',
    2: 'e',
    3: 'c',
    4: 'b',
    5: 'h',
    6: 'v',
    7: 'z',
    8: '2',
    9: 'x',
    10: 'g',
    11: 'm',
    12: 'r',
    13: 'u',
    14: 'p',
    15: 's',
    16: 'd',
    17: 'n',
    18: '6',
    19: 'k',
    20: 't'
    }

def load_model():
    checkpoint = torch.load("./pages/data/12_Captcha_Processing/captcha_model.ckpt", map_location=torch.device('cpu'))
    model = torchvision.models.resnet50(pretrained=False)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=1024, bias=True),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(in_features=1024, out_features=21 * 6, bias=True),
        )
    state_dict =checkpoint['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if 'resnet50.'in k:
            k = k.replace('resnet50.', '')
            new_state_dict[k]=v

    model.load_state_dict(new_state_dict)
    return model

def single_predict(model, image, decoding_dict, device="cpu"):

    img = transforms.Compose(
        [
            transforms.Resize((50, 250)),
            transforms.CenterCrop((50, 250)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.7570), (0.3110))
        ]
    )(image)

    if img.dim() == 3:
        img = img.unsqueeze(0)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        out = model(img.to(device))

    label = []
    encoded_vector = out.reshape(21, 6).argmax(0)
    for key in encoded_vector.detach().cpu().numpy():
        label.append(decoding_dict[key])
    return "".join(label)
try:
    if st.session_state["is_load_captcha"] == True:
        print('Đã load model')
except:
    st.session_state["is_load_captcha"] = True
    with st.spinner('Loading model..'):
        st.session_state["captcha_model"] = load_model()

file = st.file_uploader("Upload a captcha image", type=["jpg", "jpeg", "png", "tif", "bmp"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image)
    prediction = single_predict(st.session_state["captcha_model"], image, decoding_dict)
    st.subheader(f"Mã CAPTCHA nhận dạng được : :blue[{prediction}]")
