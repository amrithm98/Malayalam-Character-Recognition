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

<br><br>
<b>Training Results</b>
<br>================
<br><br>
<b>Accuracy & Sample Predictions</b>
<br>============================
<br>
Trained the model using floydhub service
<br>
Results are below :
<br>
<p align="center">
  <img style="width:100%" src=https://github.com/amrithm98/Malayalam-Character-Recognition/blob/master/DOCUMENTS_RELATED/DDD/Related_TEX_AND_IMAGES/result.png?raw=true">
</p>
                                                                                                                             <br>                       
<p align="center">
  <img src="https://github.com/amrithm98/Malayalam-Character-Recognition/blob/master/DOCUMENTS_RELATED/DDD/Related_TEX_AND_IMAGES/samp1.png?raw=true">
</p>
                                                                                                                             <br>
<p align="center">
  <img src="https://github.com/amrithm98/Malayalam-Character-Recognition/blob/master/DOCUMENTS_RELATED/DDD/Related_TEX_AND_IMAGES/samp2.png?raw=true">
</p>

<br><br>
<b>Model Used</b>
<br>==========
<br>
<p align="center">
  <img src="https://github.com/amrithm98/Malayalam-Character-Recognition/blob/master/DOCUMENTS_RELATED/DDD/Related_TEX_AND_IMAGES/model.png?raw=true">
</p>

<br><br>
<b>Deployment</b>
<br>==========
<br>
<img src="https://github.com/amrithm98/Malayalam-Character-Recognition/blob/master/DOCUMENTS_RELATED/DDD/Related_TEX_AND_IMAGES/deployment.png?raw=true">
</p>

<br><br>
<b>Coming Soon</b>
<br>===========
<br>
<ol>
  <li> Adding more data + augmentation</li>
  <li> Deployment in a web-app</li>
</ol>
