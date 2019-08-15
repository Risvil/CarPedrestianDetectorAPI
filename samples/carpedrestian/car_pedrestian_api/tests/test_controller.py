import pathlib
import io
from PIL import Image
import cv2
import numpy as np


def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200


def test_image_prediction_endpoint(flask_test_client):
    # Given
    data_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent / 'images'
    image_name = '2502287818_41e4b0c4fb_z.jpg'
    image_fullpath = data_dir / image_name

    with open(image_fullpath, 'rb') as image_file:
        file_bytes = image_file.read()
        data = dict(
            file=(io.BytesIO(bytearray(file_bytes)), image_name)
        )

    # When
    response = flask_test_client.post('/v1/detect/image',
                                      content_type='multipart/form-data',
                                      data=data
    )

    # Then
    print("Response:")
    print(response.status_code)

    assert response.status_code == 200
    print("Data:")
    print(type(response.data))

    with open('image_result.jpg', 'wb') as out_image:
        out_image.write(response.data)

    rgb_object = Image.open(io.BytesIO(response.data)).convert('RGB')
    output_image = np.asarray(rgb_object)

    cv2.imshow("Response", output_image)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
