import os
# import cv2
# import mediapipe
import ffmpeg
import pickle

def preprocess(video, output, fps=12, width=720, height=1280, exercise='squats'):
    stream = ffmpeg.input(video)
    stream = stream.filter('fps', fps=fps, round='up')
    
    if exercise == 'squats':
        out = "processed/" + output
    elif exercise == 'plank':
        out = "plank_processed/" + output
    elif exercise == 'bodywave':  # Adding support for bodywave
        out = "bodywave_processed/" + output

    stream = ffmpeg.output(stream, out)
    ffmpeg.run(stream, quiet=True)

if __name__ == "__main__":
    # Assuming you have a directory for bodywave_raw and bodywave_processed
    exercises = {
        'plank': {
            'raw': 'plank_raw',
            'processed': 'plank_processed'
        },
        'bodywave': {  # Adding support for bodywave
            'raw': 'bodywave_raw',
            'processed': 'bodywave_processed'
        }
    }

    for exercise, paths in exercises.items():
        processed = sorted(os.listdir(paths['processed']))
        raw = sorted(os.listdir(paths['raw']))

        if len(processed) == 0:
            count = 0
        else:
            count = int(processed[-1][:3])
            count += 1

        print(count)

        for i in raw:
            file = f"{paths['raw']}/{i}"
            leading_count = str(count).zfill(3)
            name = f"{leading_count}_{exercise}.mp4"
            preprocess(file, name, exercise=exercise)

            count += 1
            print(name)

            os.remove(file)
