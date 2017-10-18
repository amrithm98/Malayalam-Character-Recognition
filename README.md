# Malayalam-Character-Recognition

<b>Refer to Project-Notebooks folder for updates</b>
<br><br>
<b>Project-Notebooks</b>
<br>=================
<br>
<li>Dataset_Preparation : Converts images to 32X32 Grayscale images with <a href="https://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html">adaptive thresholding</a></li>
<li>NP-Dataset : X.npy and y.npy are numpy arrays of (34553,32,32) & (34553,) dimensions</li>
<li>Training_Dataset_Generation: Functions to Reshape and Normalize data.OneHotEncode Labels etc </li>
<li>Models/ : Folder which contains weights of trained Models</li>
<li>CNN_Model : Model is designed here. Entire training pipeline happens here. Weights are saved in the end</li>
<li>Prediction_Estimation : Generated the Prediction. Used for performance evaluation,Plotting etc</li>
