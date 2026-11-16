"""Analyze English text with the IBM Watson NLP EmotionPredict model."""

import os

import requests

WATSON_URL = (
    'https://sn-watson-emotion.labs.skills.network/v1/'
    'watson.runtime.nlp.v1/NlpService/EmotionPredict'
)
HEADERS = {'grpc-metadata-mm-model-id': 'emotion_aggregated-workflow_lang_en_stock'}
EMOTIONS = ('anger', 'disgust', 'fear', 'joy', 'sadness')


class WatsonServiceError(RuntimeError):
    """The Watson service could not provide a usable prediction."""


def emotion_detector(text_to_analyze):
    """Return the five Watson scores and the emotion with the highest score."""
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return dict.fromkeys((*EMOTIONS, 'dominant_emotion'))

    try:
        response = requests.post(
            os.environ.get('WATSON_EMOTION_URL', WATSON_URL),
            json={'raw_document': {'text': text_to_analyze}},
            headers=HEADERS,
            timeout=(5, 15),
        )
        if response.status_code == 400:
            return dict.fromkeys((*EMOTIONS, 'dominant_emotion'))
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WatsonServiceError('Watson NLP did not return a prediction.') from exc

    try:
        prediction = response.json()['emotionPredictions'][0]['emotion']
        scores = {emotion: float(prediction[emotion]) for emotion in EMOTIONS}
        if not all(0 <= score <= 1 for score in scores.values()):
            raise ValueError('Emotion score outside the expected range')
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise WatsonServiceError('Watson NLP returned invalid emotion scores.') from exc

    return {**scores, 'dominant_emotion': max(scores, key=scores.get)}
