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
  <p>For training this model I am using pytorch library. It provides torchvision module which will be used to download the pretrained model on Imagenet. I will describe the main steps in subsequents points.<a href="https://github.com/KaziShawon/imageclassification-docker-fastapi/blob/main/vgg16_vision_multiclass.ipynb"> Find the notebook here</a></p>
  <ol>
    <li>The pretrained model is downloaded, as I am having 10 classes finetuning is done to classify the desired 10 classes, it can be seen that at the very last layer it has 10 FC layers. The classes are dog,horse,elephant,butterfly,gallina,chicken,cat,cow,sheep,spider,squirrel. <br> <img src="https://i.ibb.co/xqBgqY2/getmodel.jpg" alt="getmodel" border="0"></li>
    <li>Data augmentation like: RandomRotation, RandomResizedCrop, RandomHorizontalFlip is being used to create data loader. The images are resized into 224,224 as height and width. For training set 80% of the images being used. And for training and validation set 10% per each is being used.</li>
    <li>Loss function CrossEntropyLoss is used. Adam optimizer is used to update learning parameters of each neural network parameter during during. CosineAnnealingLR is used to set learning rate accoring to the validation loss, if the loss increases the model assigns bigger step to gradient descent, for lower loss vice versa.<br> <img src="https://i.ibb.co/d2j1R7S/training.jpg" alt="training" border="0"></li>
    <li>For gradient updates in each iteration of training opt.zero_grad(), loss.backward(), opt.step() is being used. We are training our model for five epochs.</li>
    <li>Few things are important in pytorch. When the model is traing, the datset and model should be sent to the device (cpu/gpu). And before training model should be set to train model with model.train(). For evaluation and test model.eval(). When the evaluation and test phase happens it should be run with torch.no_grad(), as it obstructs the model to update the gradients at that time.</li>
    <li>Softmax function is used for probability of multiclass classification.</li>
    <li>The model has achieved 96% accuracy with test set, which dataset model has not seen during training.</li>
    <li>The best model according to the best validation accuracy is being saved in directory.</li>
  </ol>
</div>
<div>
  <h2>Creating an API and making prediction on a local server</h2>
  We will use fastapi to make a classification on local server, then we will build docker image out of it and make a prediction with curl command.
  <ol>
    <li>A Classification module in classifcation.py file is being set up. Where the pretrained model is being downloaded one time and the required finetuning is done according to the model's architecture. Every image being sent as a request it will be resized to our model's desired input.</li>
    <li>In server.py the get and post method is built. The requested image will be read as bytes format and for every request our predict function will called to return the prediction as well as confidence. <br><img src="https://i.ibb.co/XxH7tGw/uvicorn.jpg" alt="uvicorn" border="0"> <br><img src="https://i.ibb.co/yhrmZBz/fastapiinfernec.jpg" alt="fastapiinfernec" border="0"></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
  </ol>
</div>
