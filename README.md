# The end-to-end AI framework for Arabic handwritten legal amounts recognition

  
The end-to-end proposed AI framework for Arabic handwritten legal amounts recognition. 

The proposed framework comprises three stages: word detection, word classification, and courtesy amount conversion.

This project has been trained and tested on .
  


  
## Workflow

The project works in three phases.
![ui app.png](https://github.com/Hakim-Abdo/ArabicHandwrittenLegalAmountToCourtesyAmount/ui_app.png)

### [Legal Word Detection](Training/YOLOv5/)

Once the legal amount image is uplade and and convert button is clicked by user a [YOLOv5](https://github.com/ultralytics/yolov5) model will be run to detect and crop the the words present in the image. YOLO model is trained using private dataset .

### [recognize croped words and convert them into courtesy amount](training /hybrid CNNs-ViT model) 

the croped word images will pass into hybrid CNNs-ViT model for recogize them;The hybrid CNNs-ViT model is trained using  private dataset.

### [genrate the courtesy amount](LegalToCourtesy algorithm) 

then the recogized word will pass as list into LegalToCourtesy algorithm for genrate the courtesy amount.



## RUN THE UI APP
 
To run the app, `streamlit run ArabicHandwrittenLegalAmountToCourtesyAmount_app.py   `




## Folder Structure
`ArabicHandwrittenLegalAmountToCourtesyAmount_app.py`  
`ArabicLegalAmountWordDetection_WithYOLOv5s.ipynb`
`Ensemble_CNN _ViT_ForArabicWordRecognition.ipynb`
`LegaltoCourtesyAmount.py`
&nbsp;&nbsp;&nbsp;&nbsp; |-> `models`   
&nbsp;&nbsp;&nbsp;&nbsp; |-> `runs`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `OutputWords`  
&nbsp;&nbsp;&nbsp;&nbsp; |-> `uploaded_file`   


### models
Contains the trained models.  

  

### runs\OutputWords
Stores the results of croped words by YOLOv5 .  
* YOLOv5 results are stored `yolov5` runs\OutputWords. A new folder with name of uploaded_file' is created every time the model is run.   
