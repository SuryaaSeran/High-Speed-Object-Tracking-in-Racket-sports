base_headers = {"User-Agent": "Mozilla/5.0", "accept-language": "en-US,en"}

import queue
import numpy as np
from PIL import Image, ImageDraw
import keras
from keras.models import *
from keras.layers import *
from flask import request
from flask import jsonify
from flask import Flask
import pytube
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/hello',methods=['POST'])
def hello():
    message = request.get_json(force=True)
    name = message['name']
    video_path = output(name)
    response = {
        'output_video_path': video_path, 'status': 1
    }
    print(response)
    return jsonify(response)

def SegNet(input_shape=(9,360, 640), classes=256):
    img_input = Input(shape=input_shape)
    x = img_input
    # Encoder
    x = Conv2D(64, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2,2), data_format='channels_first', padding="same")(x)

    x = Conv2D(128, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(128, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2,2), data_format='channels_first', padding="same")(x)

    x = Conv2D(256, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(256, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(256, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2,2), data_format='channels_first', padding="same")(x)

    x = Conv2D(512, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(512, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(512, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    
    # Decoder
    
    x = Conv2D(512, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(512, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(512, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2),  data_format='channels_first')(x)
    x = Conv2D(256, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(256, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(128, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2),  data_format='channels_first')(x)
    x = Conv2D(128, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(64, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2),  data_format='channels_first')(x)
    x = Conv2D(64, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(classes, 3, padding="same", kernel_initializer='random_uniform', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #x = Conv2D(classes, 1, 1, padding="valid")(x)
    o_shape = Model(img_input, x).output_shape
    print("output shape:", o_shape[1], o_shape[2], o_shape[3])
    # layer24 output shape: 256, 360, 640

    OutputHeight = o_shape[2]
    OutputWidth = o_shape[3]

    # reshape the size to (256, 360*640)
    x = (Reshape((-1, OutputHeight * OutputWidth)))(x)

    # change dimension order to (360*640, 256)
    x = (keras.layers.Permute((2, 1)))(x)

    gaussian_output = (Activation('softmax'))(x)

    model = Model(img_input, gaussian_output)
    model.outputWidth = OutputWidth
    model.outputHeight = OutputHeight

    # show model's details
    model.summary()

    return model

def download(link):
    video_url = str(link)
    youtube = pytube.YouTube(video_url)
    video = youtube.streams.get_by_resolution('720p')
    video.download(output_path='sample_video/Tennis/', filename='input')


def output(link):
    download(link)
    input_video_path = "sample_video/Tennis/input.mp4"
    output_video_path_1 = "sample_video/Tennis/output1.mp4"
    output_video_path_2 = "sample_video/Tennis/output2.mp4"
    save_weights_path = "sample_video/Ver7/segnet_weights_5.0"
    n_classes = 256

    if output_video_path_1 == "":
        #output video in same path
        output_video_path = input_video_path.split('.')[0] + "_TrackNet.mp4"

    #get video fps&video size
    video = cv2.VideoCapture(input_video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #start from first frame
    currentFrame = 0

    #width and height in TrackNet
    width , height = 640, 360
    img, img1, img2 = None, None, None

    m=SegNet()
    m.compile(loss='categorical_crossentropy', optimizer= 'adadelta' , metrics=['accuracy'])
    m.load_weights(  save_weights_path  )
    print("Model weights read")

    # In order to draw the trajectory of tennis, we need to save the coordinate of preious 7 frames
    q = queue.deque()
    for i in range(0,10):
        q.appendleft(None)

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_video = cv2.VideoWriter(output_video_path_1,fourcc, fps, (output_width,output_height))


    #both first and second frames cant be predict, so we directly write the frames to output video
    #capture frame-by-frame
    video.set(1,currentFrame);
    ret, img1 = video.read()
    #write image to video
    output_video.write(img1)
    currentFrame +=1
    #resize it
    img1 = cv2.resize(img1, ( width , height ))
    #input must be float type
    img1 = img1.astype(np.float32)

    #capture frame-by-frame
    video.set(1,currentFrame);
    ret, img = video.read()
    #write image to video
    output_video.write(img)
    currentFrame +=1
    #resize it
    img = cv2.resize(img, ( width , height ))
    #input must be float type
    img = img.astype(np.float32)



    while(True):

        img2 = img1
        img1 = img

        #capture frame-by-frame
        video.set(1,currentFrame);
        ret, img = video.read()

        #if there dont have any frame in video, break
        if not ret:
            break

        #img is the frame that TrackNet will predict the position
        #since we need to change the size and type of img, copy it to output_img
        output_img = img

        #resize it
        img = cv2.resize(img, ( width , height ))
        #input must be float type
        img = img.astype(np.float32)


        #combine three imgs to  (width , height, rgb*3)
        X =  np.concatenate((img, img1, img2),axis=2)

        #since the odering of TrackNet  is 'channels_first', so we need to change the axis
        X = np.rollaxis(X, 2, 0)
        #prdict heatmap
        pr = m.predict( np.array([X]) )[0]

        #since TrackNet output is ( net_output_height*model_output_width , n_classes )
        #so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
        #.argmax( axis=2 ) => select the largest probability as class
        pr = pr.reshape(( height ,  width , n_classes ) ).argmax( axis=2 )

        #cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
        pr = pr.astype(np.uint8)

        #reshape the image size as original input image
        heatmap = cv2.resize(pr  , (output_width, output_height ))

        #heatmap is converted into a binary image by threshold method.
        ret,heatmap = cv2.threshold(heatmap,127,255,cv2.THRESH_BINARY)

        #find the circle in image with 2<=radius<=7
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=50,param2=2,minRadius=2,maxRadius=7)

        #In order to draw the circle in output_img, we need to used PIL library
        #Convert opencv image format to PIL image format
        PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(PIL_image)

        #check if there have any tennis be detected
        if circles is not None:
            #if only one tennis be detected
            if len(circles) == 1:

                x = int(circles[0][0][0])
                y = int(circles[0][0][1])
                print(currentFrame, x,y)

                #push x,y to queue
                q.appendleft([x,y])
                #pop x,y from queue
                q.pop()
            else:
                #push None to queue
                print("Multiple circles")
                q.appendleft(None)
                #pop x,y from queue
                q.pop()
        else:
            #push None to queue
            print("No circles")
            q.appendleft(None)
            #pop x,y from queue
            q.pop()

        #draw current frame prediction and previous 7 frames as yellow circle, total: 8 frames
        for i in range(0,10):
            if q[i] is not None:
                draw_x = q[i][0]
                draw_y = q[i][1]
                bbox =  (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                draw = ImageDraw.Draw(PIL_image)
                draw.ellipse(bbox, outline ='yellow')
                del draw

        #Convert PIL image format back to opencv image format
        opencvImage =  cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
        #write image to output_video
        output_video.write(opencvImage)

        #next frame
        currentFrame += 1

    # everything is done, release the video
    video.release()
    output_video.release()
    print("finish")
    return output_video_path_1


