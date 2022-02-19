"""An example of generating a gif explanation for a self-created image of a digit (similar to MNIST). """
import argparse
import os
from os.path import exists, dirname
import sys

parent_dir = dirname(os.path.abspath(os.getcwd()))
sys.path.append(parent_dir)

from bayes.explanations import BayesLocalExplanations, explain_many
from bayes.data_routines import get_dataset_by_name
from bayes.models import *
from image_posterior import create_gif	

parser = argparse.ArgumentParser()
parser.add_argument("--cred_width", type=float, default=0.1)
parser.add_argument("--save_loc", type=str, required=True)
parser.add_argument("--n_top_segs", type=int, default=5)
parser.add_argument("--n_gif_images", type=int, default=20)

IMAGE_NAME = "self_mnist5"
CLASS = 5


def get_image_data():
    """Gets the image data and model."""
    img = get_dataset_by_name(IMAGE_NAME, get_label=False)
    model_and_data = process_mnist_get_model(img)
    return img, model_and_data


def main(args):
    img, model_and_data = get_image_data()

    # Unpack data
    xtest = model_and_data["xtest"]
    ytest = model_and_data["ytest"]
    segs = model_and_data["xtest_segs"]
    get_model = model_and_data["model"]
    label = model_and_data["label"]

    # Unpack instance and segments
    instance = xtest[0]
    segments = segs[0]

    # Get wrapped model
    cur_model = get_model(instance, segments)

    # Get background data
    xtrain = get_xtrain(segments)

    prediction = np.argmax(cur_model(xtrain[:1]), axis=1)
    assert prediction == CLASS, f"Prediction is {prediction} not {CLASS}"

    # Compute explanation
    exp_init = BayesLocalExplanations(training_data=xtrain,
                                              data="image",
                                              kernel="lime",
                                              categorical_features=np.arange(xtrain.shape[1]),
                                              verbose=True)
    rout = exp_init.explain(classifier_f=cur_model,
                            data=np.ones_like(xtrain[0]),
                            label=CLASS,
                            cred_width=args.cred_width,
                            focus_sample=False,
                            l2=False)

    # Create the gif of the explanation
    create_gif(rout['blr'], segments, instance, args.save_loc, args.n_gif_images, args.n_top_segs)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
