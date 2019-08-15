import cv2
from tqdm import tqdm
from car_pedrestian_model import CarPedrestianModel, merge_results_on_image


####################################
### Segment and visualize image
####################################
def segment_image(model, image_name):
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = model.detect_and_draw_results(image, verbose=0)

    cv2.imshow('Segmented image', image)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    return image



####################################
### Segment video and save to disk
####################################
def segment_video(model, video_name, show_video=False):
    cap = cv2.VideoCapture(video_name)

    if cap.isOpened():
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_width = width
        out_height = height

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (out_width, out_height))
        
        for i in tqdm(range(num_frames)):
            ret, frame = cap.read()

            if ret:
                frame = model.detect_and_draw_results(frame, verbose=0)

                # write to disk
                out.write(frame)

                if show_video:
                    cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        out.release()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = CarPedrestianModel()

    image_name = "../../images/2502287818_41e4b0c4fb_z.jpg"
    segment_image(model, image_name)

    video_name = "../../videos/russian_traffic.mp4"
    segment_video(model, video_name, show_video=True)
