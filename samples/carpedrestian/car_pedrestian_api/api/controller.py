from flask import Blueprint, request, jsonify, send_file

import os
import sys
import pathlib
from werkzeug.utils import secure_filename

from api.config import get_logger, UPLOAD_FOLDER
from api.validation import allowed_file
from api import __version__ as api_version


MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(MODEL_PATH))
from supermodelmaximum.car_pedrestian_model import make_single_prediction



import cv2


_logger = get_logger(logger_name=__name__)


prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'OK'


@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': 0, 'api_version': api_version})


@prediction_app.route('/v1/detect/image', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        # Step 1: Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify('No file found'), 400

        file = request.files['file']

        # Step 2: Basic file extension validation
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # Step 3: Save the file
            # Note, in production this would require careful
            # validation, management and clean up

            # TODO: Make predictions in memory, do not store the image
            file.save(os.path.join(UPLOAD_FOLDER, filename))

            _logger.debug(f'Inputs: {filename}')

            # Step 4: perform prediction
            result = make_single_prediction(
                image_name=filename,
                image_directory=UPLOAD_FOLDER
            )

            # 4. Remove input image from disk
            full_filename = os.path.join(UPLOAD_FOLDER, filename)
            os.remove(full_filename)


            # TODO: Save image to disk, then use "send_image"
            # or io.BytesIO to send the image as a byte array

        #     _logger.debug(f'Outputs: {result}')

        # readable_predictions = result.get('readable_predictions')
        # version = result.get('version')

            # _logger.warning("Result:")
            # _logger.warning(result)

            out_filename = str(UPLOAD_FOLDER / 'output.jpg')
            cv2.imwrite(out_filename, result)
            return send_file(out_filename, mimetype='image/jpg')

        return jsonify(
            {'error': 'No suitable image received'}
        )
        
        # # Step 5: Return the response as JSON
        # return jsonify(
        #     {
        #         'readable_predictions': readable_predictions[0],
        #         'version': version
        #     }
        # )
