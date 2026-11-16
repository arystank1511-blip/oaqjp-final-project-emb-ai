# Final project

An emotion detection web app for IBM's **Developing AI Applications with Python
and Flask** course. It uses the Watson NLP EmotionPredict model to score anger,
disgust, fear, joy, and sadness in English text and reports the highest score's
emotion. The application also handles blank input and service errors.

## Run the app

With Python 3.10 or newer, run:

```sh
python -m pip install -r requirements.txt
python server.py
```

Open port 5000 in the Skills Network Cloud IDE, or visit `http://127.0.0.1:5000/`
when running locally. The default Watson endpoint belongs to the course lab
network. Outside the lab it may be unreachable. To use a different deployment
of the same Watson model, set `WATSON_EMOTION_URL` to its full EmotionPredict URL.

## Verify

```sh
python -m pip install -r requirements-dev.txt
python -m unittest -v
python -m pylint server.py EmotionDetection test_emotion_detection.py
```

The unit tests substitute Watson's documented HTTP responses; they do not need
network access. Inside the lab, run `RUN_WATSON_LIVE=1 python3 -m unittest -v`
to also verify the five sample sentences against the live model. You can also
exercise the model directly:

```python
from EmotionDetection import emotion_detector
print(emotion_detector('I am so happy I am doing this'))
```
