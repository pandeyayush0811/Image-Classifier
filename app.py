from fastai.vision.all import *
import gradio as gr

learn = load_learner('model.pkl')

def classify_image(img):
    pred_class, pred_idx, outputs = learn.predict(img)
    return str(pred_class)

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.inputs.Image(type="pil"),
    outputs="text",
    title="Image Classifier",
    description="Upload an image of a Cat, Dog, Car and Bike and the model will classify it!"
)

if __name__ == "__main__":
    iface.launch()
