import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
    model.eval()
    return model, device

st.title("Handwritten Digit Recognition")
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=200,
    width=200,
    drawing_mode="freedraw"
)

if st.button("Predict"):
    model, device = load_model()
    if canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data.astype("uint8"))
        image = ImageOps.grayscale(image)
        image = image.resize((28, 28))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        input_img = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_img)
            _, prediction = torch.max(output, 1)
        st.write("Predicted Digit:", prediction.item())