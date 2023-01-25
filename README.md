# License plate recognition template

## Project descritpion

This is the license plate recognition model of Group 18 for Image Processing. The goal of this project is to create a pipeline that can localize and recognize European car plates. Most of the data that this model will be tested on are Dutch license plates, but some other European license plate may also need to be tested. 

## Quick pipeline explanation

The overall pipeline is separated into 2 stages - localization and recognition, wherein a third optional one may follow, namely a majority voting stage. 

### Localization

For localization, we run the input video through multiple masks to color segment the frame input. Once that is done, we find all contours on the image, filter them based on a few conditions to make a distinction between license plates and other objects. If no plates were located, then memoization takes place to output the previously located plates, with the assumption that the scenery has not changed.

### Recognition

We denoise the license plate and crop the image in all its directions to leave only the letters. We then segment the characters by scanning through the image horizontally to find character jumps, or the best places to cut up the image. We concatenate all the character matches on an individual basis.

### Majority voting

An optional step for evaluating the accuracy of the model. We accumulate a cache to store all the guesses and then we output a label when a frame changes significantly, on the basis of basic majority voting.

**Of course, each stage is not as simple as that, a more thorough picture of the model may be garnered from reading the comments, or looking through the team's posters for each stage.**

### Cross-validation

The project also has a cross-validation set up for localization and recognition separately, to fine tune the hyperparameters that are used for the two stages. The localization cross-validation takes a certain train video, splits it by samples, and compares bounding boxes of the license plates. The recognition cross-validation takes a provided directory of cropped license plates and compares the recognized registration number to the image's filename. For both localization and recognition the codebase incorporates a way to quickly gather training samples by clicking on the train videos, the type of data collection being toggleable by keycuts. Some other keycuts are added to ease the pausing and fast-forwarding the preview of a train video.

## How to run

Simply run the provided `evaluator.sh` script, which finds a video to be tested, outputs a CSV file containing all final predictions, and are evaluated over the `evaluation.py` script. The `evaluator_that_works.sh` is just a slightly modified version of the original evaluator that excludes some results for the test video find operation, specifically the git, cache, venv, and pytest folders and files. If the evaluator scripts do not work, then delete files or restructure the project such that the find operation finds only a single match in the project directory tree. The regex that is used is the default (`'*test*'`).
