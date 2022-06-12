from flask import Flask, jsonify, request
import json
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
from base64 import b64decode, b64encode
from io import BytesIO

app = Flask(__name__)

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
tf.saved_model.LoadOptions(
    experimental_io_device="/job:localhost",
)

detector = hub.load(module_handle).signatures["default"]


@app.route(
    '/predict',methods=['GET']
)



def predict():
    event = json.loads(request.data)
    values = event['image']
    decoded_image = b64decode(values)
    img = Image.open(BytesIO(decoded_image))
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    converted_img = input_tensor[:, :, :, :3]
    converted_img2 = tf.image.convert_image_dtype(converted_img, tf.float32)
    result = detector(converted_img2)
    result = {key:value.numpy() for key,value in result.items()}
    tup = list(zip(result["detection_class_entities"], result["detection_scores"]))
    print(tup[0])
  
    image = image_np
    boxes = result["detection_boxes"]
    class_names = result["detection_class_entities"]
    scores = result["detection_scores"]
    max_boxes=1
    min_score=0.1
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                      25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                              int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            print("color dtype", type(color))
            print(color)
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            
            thickness=4
            display_str_list=[display_str]
            #"""Adds a bounding box to an image."""
            draw = ImageDraw.Draw(image_pil)
            im_width, im_height = image_pil.size
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
            draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                 (left, top)],
                width=thickness,
                fill=color)

            # If the total height of the display strings added to the top of the bounding
            # box exceeds the top of the image, stack the strings below the bounding box
            # instead of above.
            display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
            # Each display_str has a top and bottom margin of 0.05x.
            total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

            if top > total_display_str_height:
                text_bottom = top
            else:
                text_bottom = top + total_display_str_height
            # Reverse list and print from bottom to top.
            for display_str in display_str_list[::-1]:
                text_width, text_height = font.getsize(display_str)
                margin = np.ceil(0.05 * text_height)
                draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                    fill=color)
                draw.text((left + margin, text_bottom - text_height - margin),
                display_str,
                fill="black",
                font=font)
                text_bottom -= text_height - 2 * margin
        np.copyto(image, np.array(image_pil))

    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)
    data = Image.fromarray(image)
    print(data)

    #Decoding the image into base64 start ----------------------------------------------------------------------------
    encoded_image = b64encode(image).decode('utf8')
    # Decoding the image into base64 end ----------------------------------------------------------------------------

    return jsonify({"Object": str(tup[0][0]),"Accuracy":str(tup[0][1]), "Image":encoded_image})
    #print({"request": tup[0], "Image":encoded_image})






if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000')





