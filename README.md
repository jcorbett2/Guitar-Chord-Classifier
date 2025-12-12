# Guitar-Chord-Classifier
This is the Final for my AI 3300 class. It is a guitar chord classifier


# Description
This project is split into 3 main sections. The first is the classifier model which takes an audio input from the user and outputs the chord being played in the audio clip. This is limited to A-G including sharps and all major and minor chords, 24 chords in total. The architechture used for this section is a convolutional neural network and it is trained on chroma cqt features from the audio files to classify the different chords.

The second part is the music theory logic which tells the user, based on the chords played, what key that combination of chords is most likely in. 

This is then given to the third part of the project which is another AI model trained to give the user a commonly used chord progression based on the chords being played in that key. The architecture for this model is a type of recurrent neural network called an LSTM, or a long short-term memory model. It is trained on a small dataset of common chord progressions to predict the next token in the sequence given a list of chords.



# How to Use
To get started there is a requirements.txt file that the user can use to download all the necessary dependencies for this project.


## Create Virtual Environment:

Python3 -m venv venv_name



## Enter the virtual Environment:

source venv_name/bin/activate



## Download dependencies:

pip install requirements.txt



## To run the project and start classifying chords:

Python main.py ./user_input/"file_name"



This can be any directory but the "user_input" directory contains all the test files I used and contains plenty of options for the user to play around with. If you would like to test your own audio clips you can do this by creating a folder on the repository and adding your own .wav files to it!



## Structure of user_input:

user_input/

    2_chord_clips/

        test1/

            test1-1.wav

            test1-2.wav

    3_chord_clips/
    
    4_chord_clips/


Each of the 3 directories in user_input contain 6 tests for the 3 different numbers of chord tests. All audio clips used to train and test the model were recorded by me.


