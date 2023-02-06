
import numpy as np
from PIL import Image
from SOLAR.searching_global import search_solar
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import cv2
import base64

app = Flask(__name__)

# Read image features
fe = search_solar()
features = []
img_paths = []
# for feature_path in Path("./static/feature").glob("*.npy"):
#     features.append(np.load(feature_path))
#     img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)
img_DB = "/Users/sonnguyen/Desktop/AI/DATK/data_pill/train/"

def convert_bas64(img_path):
    img = cv2.imread(img_path)
    _, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img'].read()
        # read query image
        np_image = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        path_save_query = "/Users/sonnguyen/Desktop/AI/DATK/DATN_web/query_path_save/" + request.files['query_img'].filename

        cv2.imwrite(str(path_save_query), img)
        # path_save_query = path_save_query.stem
        # Run search
        img_path_search, scores  = fe.searching_img(query_path= img)
        # print(img_path_search)
        scores = [(scores[id], img_DB + img_path_search[id]) for id in range(len(scores))]
        # binary_images = [cv2.imencode(".jpg", img_path_search)[1].tobytes() for retrieved_image in retrieved_images]

        return render_template('index.html',
                               query_path=path_save_query,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    # app.run("0.0.0.0")
    app.run(host='127.0.0.1', threaded=True)