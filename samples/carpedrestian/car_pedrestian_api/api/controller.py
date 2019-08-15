from flask import Blueprint, request, jsonify
#from car_pedrestian_model.car_pedrestian_model import 
import os
from werkzeug.utils import secure_filename

from api.config import get_logger, UPLOAD_FOLDER
from api.validation import allowed_file
from api import __version__ as api_version


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


@prediction_app.route('/v1/predict/image', methods=['POST'])
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

            _logger.debeug(f'Outputs: {result}')

        readable_predictions = result.get('readable_predictions')
        version = result.get('version')
        
        # Step 5: Return the response as JSON
        return jsonify(
            {
                'readable_preedictions': readable_predictions[0],
                'version': version
            }
        )
