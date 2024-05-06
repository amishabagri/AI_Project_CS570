Please install the following dependencies

numpy
cv2
os
dlib
tensorflow
keras
scikit-learn


If you are using MacOS you need to install cmake and dlib

pip install cmake
pip install dlib

For starting the program:

python create_data.py
Then in the terminal write your name and press enter

python face_recognise.py
It will train and recognize the face

python evaluation.py
It will give the model evaluation

In the create_data, I am detecting the face using HAAR Cascade and storing the captured images.
In the face_recognise.py, I am training the model and storing it in a .h5 file and using the model I am recoginising the face. It will also perform liveness test which uses depth and eye blibking critera to detect of the frame captured is of actual person or not
In the evaluation.py I am calculating the test loss, test accuracy, confusion matrix and F1 accuracy.

P.S Please run the face_recognise.py before evaluation. 

Thank you
