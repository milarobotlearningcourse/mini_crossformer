

# HW: Making a Generalist Robotics Policy

A Generalist Robtoics Policy is made up from a modified vision transformer. I vision transfer is a modified version of a transformer that is designed to process images instead of text. In order for a transformer to process images the images need to be sliced up into patches that can be tokenized.

Tasks in this homework:

## Finish the GRP transformer code following the design from the [octo paper](https://octo-models.github.io/).

The provided code is an example of a vision transformer. Modifiy this code to become a multi-modal transformer modal that accepts images and text as input. Make sure to impliment the block masking to train the model to work when goals are provided via images or text.


### Discrete vs Continuous Action Space

There are different methods that can be used to describe the action distribution. Many papers have discretetized the action space (OpenAI Hand), resulting in good performance. Train the GRP model with a discrete representation vs a continuous representation and compare the perfroamnce of these two distributions. Compare the performance in [simpleEnv](https://simpler-env.github.io/)

### Repalce the text encoder with the one from T5

The text tokenization and encoding provided in the initial version of the code is very simple and basic. It may be possible to improve goal generalizing by improving the tokenization used. Use the tokenizer from the T5 model to tokenize the text used to encode the goal descriptions.


## Add another Dataset

The dataset cleaned for the original version of the homework is 64 x 64 x 3. This image resultion can work but will often cause issues when objects are too small. Becasue the objects are so small they appear in the image via very few pixels, making it challenging for the GRP to pickup on these fine details. re-create the dataset with 96 x 96 x 3 size images. Does this increase in image size improve the performance?

## State History

For most robotics problems a single image is not enough to determine the dyanmics well enough to predict the next state. This lack of dynamics information means the model can only solve certain tasks to a limited extent. To provide the model with sufficient state information update the GRP input to include 2 images from the state history and evalaute the performance of this new model.

## Action Stacking

One of the many methods used to smooth motion and compensate for multi-modal behaviour in the dataset is to predict many actions at a time. 

## Change the image patching to work for variable size images.