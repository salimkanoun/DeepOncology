

from .lib.processing_inference import *
from Models.classification_model import net_classification


def inference_classification(model_path, CT_path, output_shape, angle):
    input = preprocessing_image_to_bytes(CT_path, output_shape, angle)
    model = net_classification().double()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    output = model.forward(input)
    result = postprocessing_classification(output)
    return result

if __name__ == "__main__":
    inference_classification()