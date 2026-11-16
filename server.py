"""Serve the emotion detection page and its Flask endpoint."""

from flask import Flask, render_template, request

from EmotionDetection import emotion_detector
from EmotionDetection.emotion_detection import WatsonServiceError

app = Flask(__name__)


@app.get('/')
def index():
    """Render the form for entering English text."""
    return render_template('index.html')


@app.get('/emotionDetector')
def detect_emotion():
    """Return the scores and dominant emotion for the submitted text."""
    try:
        result = emotion_detector(request.args.get('textToAnalyze', ''))
    except WatsonServiceError:
        return 'Watson NLP is unavailable. Please try again later.', 502

    dominant = result['dominant_emotion']
    if dominant is None:
        return 'Invalid text! Please try again!', 400

    return (
        f"For the given statement, the system response is 'anger': {result['anger']}, "
        f"'disgust': {result['disgust']}, 'fear': {result['fear']}, "
        f"'joy': {result['joy']} and 'sadness': {result['sadness']}. "
        f'The dominant emotion is {dominant}.'
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
