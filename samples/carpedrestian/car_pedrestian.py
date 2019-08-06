import skimage.io
from car_pedrestian import CarPedrestianModel, merge_results_on_image

# def apply_mask(image, mask, color, alpha=0.5):
#     """Apply the given mask to the image.
#     """
#     for c in range(3):
#         image[:, :, c] = np.where(mask == 1,
#                                   image[:, :, c] *
#                                   (1 - alpha) + alpha * color[c] * 255,
#                                   image[:, :, c])
#     return image


####################################
### Segment and visualize image
####################################
def segment_image(image_name):
    image = skimage.io.imread(image_name)

    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    # ax = get_ax(1)
    r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
    #                             labels, r['scores'], ax=ax,
    #                             title="Predictions") #dataset.class_names
    image = merge_results_on_image(image, r['rois'], r['masks'], r['class_ids'])

    cv2.imshow('Segmented image', image)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    return image



####################################
### Segment video and save to disk
####################################
def segment_video(video_name, show_video=False):
    cap = cv2.VideoCapture(video_name)

    if cap.isOpened():
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    #     out_width = width // 4
    #     out_height = height // 4
        out_width = width
        out_height = height

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (out_width, out_height))
        
        for i in tqdm(range(num_frames)):
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # reduce image size for speed
    #         if width != out_width:
    #             frame = cv2.resize(frame, (out_width, out_height), interpolation = cv2.INTER_CUBIC)
           
            # Segment the image
            results = model.detect([frame], verbose=0)

            # draw bounding boxes and segmentation masks
            r = results[0]
            frame = visualize.merge_results_on_image(frame, r['rois'], r['masks'], r['class_ids'], 
                                        labels, r['scores'],
                                        title="Predictions") #dataset.class_names

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
    image_name = "../../images/2502287818_41e4b0c4fb_z.jpg"
    segment_image(image_name)

    video_name = "../../videos/russian_traffic.mp4"
    segment_video(video_name, show_video=True)
