# imageclassification-docker-fastapi
## About the Project
The target of the project is to build a image classification model and move the model to production. The approaches which are being embraced are:
<ol>
  <li>Building a deeplearning classification model from freely available datset. In this case <a href="https://www.kaggle.com/alessiocorrado99/animals10">This dataset is being used</a>.</li>
  <li>Set up a communication URL.</li>
  <li>Accept input from a wide variety of environments/formats when it is sent to the URL.</li>
  <li>Convert every form of input into the exact format that the machine learning model needs as input.</li>
  <li>Make predictions with the trained deep learning-based model.</li>
  <li>Convert predictions into the right format and respond to the client's request with the prediction.</li>
  <li>Create a Docker file for classification and deploying it to cloud instance.
  <li>Deploying the FMNIST classifier on an Amazon Web Services (AWS) Elastic Compute Cloud (EC2)</li>
</ol>

<div>
  <h2>Building a VGG16 Model</h2>
  <p>For training this model I am using pytorch library. It provides torchvision module which will be used to download the pretrained model on Imagenet. I will describe the main steps in subsequents points.<a href="https://github.com/KaziShawon/imageclassification-docker-fastapi/blob/main/vgg16_vision_multiclass.ipynb"  target="_blank"> Find the notebook here</a></p>
  <ol>
    <li>Ok</li>
  </ol>
</div>
