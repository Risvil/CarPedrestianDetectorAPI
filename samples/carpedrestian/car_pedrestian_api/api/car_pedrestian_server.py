from car_pedrestian_model import CarPedrestianModel
from flask import Flask, request, Response


app = Flask(__name__)

@app.route("/segment-image", methods=['POST'])
def segment_image():
    if request.method == 'POST':
        #flask.request.files.get('')

        image = open()

        nparr = np.fromstring(request.data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # generate segmentation with model
        segmentation = model.predict_and_draw_results(image)


        # TODO: Read image properly and then send the response back to the client



if __name__ == '__main__':
    model = CarPedrestianModel()
    app.run()