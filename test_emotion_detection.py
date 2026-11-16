"""Exercise emotion detection and the Flask interface without network access."""

import os
import unittest
from unittest.mock import Mock, patch

import requests

from EmotionDetection import emotion_detector
from EmotionDetection.emotion_detection import WatsonServiceError
from server import app


COURSE_CASES = (
    ('I am glad this happened', 'joy'),
    ('I am really mad about this', 'anger'),
    ('I feel disgusted just hearing about this', 'disgust'),
    ('I am so sad about this', 'sadness'),
    ('I am really afraid that this will happen', 'fear'),
)
EMOTIONS = ('anger', 'disgust', 'fear', 'joy', 'sadness')


def watson_response(dominant_emotion):
    """Construct a test response in Watson's documented JSON format."""
    scores = dict.fromkeys(EMOTIONS, 0.03)
    scores[dominant_emotion] = 0.88
    response = Mock(status_code=200)
    response.json.return_value = {'emotionPredictions': [{'emotion': scores}]}
    return response


@patch('EmotionDetection.emotion_detection.requests.post')
class TestEmotionDetector(unittest.TestCase):
    """Check the Watson client in isolation."""

    def test_course_examples_and_output_contract(self, post):
        """Check the mapping from a Watson response to the result dictionary."""
        for sample, dominant in COURSE_CASES:
            with self.subTest(sample=sample):
                post.return_value = watson_response(dominant)
                result = emotion_detector(sample)
                self.assertEqual(set(result), {*EMOTIONS, 'dominant_emotion'})
                self.assertEqual(result['dominant_emotion'], dominant)
                self.assertEqual(result[dominant], 0.88)
                call = post.call_args
                self.assertTrue(call.args[0].endswith('/NlpService/EmotionPredict'))
                self.assertEqual(call.kwargs['json'], {'raw_document': {'text': sample}})
                self.assertEqual(call.kwargs['headers']['grpc-metadata-mm-model-id'],
                                 'emotion_aggregated-workflow_lang_en_stock')

    def test_blank_input_returns_empty_scores_without_calling_watson(self, post):
        """Treat empty and whitespace input as invalid without sending a request."""
        for sample in ('', '   ', None):
            with self.subTest(sample=sample):
                result = emotion_detector(sample)
                self.assertEqual(result, dict.fromkeys((*EMOTIONS, 'dominant_emotion')))
        post.assert_not_called()

    def test_watson_bad_request_returns_empty_scores(self, post):
        """Translate Watson's HTTP 400 into the expected blank result."""
        post.return_value = Mock(status_code=400)
        result = emotion_detector('An input rejected by Watson')
        self.assertEqual(result, dict.fromkeys((*EMOTIONS, 'dominant_emotion')))
        post.return_value.json.assert_not_called()

    def test_watson_timeout_is_reported_as_service_failure(self, post):
        """Report a timeout separately from invalid input."""
        post.side_effect = requests.Timeout
        with self.assertRaises(WatsonServiceError):
            emotion_detector('I am glad this happened')

    def test_missing_scores_are_reported_as_service_failure(self, post):
        """Reject a response with no usable prediction."""
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {'emotionPredictions': []}
        with self.assertRaises(WatsonServiceError):
            emotion_detector('I am glad this happened')


class TestFlaskApp(unittest.TestCase):
    """Exercise the actual Flask routes in process."""

    def setUp(self):
        self.client = app.test_client()

    def test_index_loads_input_form(self):
        """Check that Flask renders the application's input form."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="textToAnalyze"', response.data)

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_result_displays_scores_and_dominant_emotion(self, post):
        """Check that the route displays five scores and preserves input text."""
        post.return_value = watson_response('joy')
        sample = 'I am glad this happened & we won!'
        response = self.client.get('/emotionDetector', query_string={'textToAnalyze': sample})
        self.assertEqual(response.status_code, 200)
        for emotion in EMOTIONS:
            self.assertIn(f"'{emotion}':".encode(), response.data)
        self.assertIn(b'The dominant emotion is joy.', response.data)
        self.assertEqual(post.call_args.kwargs['json']['raw_document']['text'], sample)

    def test_blank_form_displays_error(self):
        """Return the lab's invalid input message with HTTP 400."""
        response = self.client.get('/emotionDetector', query_string={'textToAnalyze': ' '})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.text, 'Invalid text! Please try again!')

    @patch('EmotionDetection.emotion_detection.requests.post', side_effect=requests.Timeout)
    def test_watson_failure_is_not_reported_as_blank_input(self, _post):
        """Explain a Watson outage to the caller with HTTP 502."""
        response = self.client.get('/emotionDetector', query_string={'textToAnalyze': 'Hello'})
        self.assertEqual(response.status_code, 502)
        self.assertIn('Watson', response.text)


class LiveWatsonChecks(unittest.TestCase):
    """Check every required course example against the live lab model."""

    def test_course_classifications(self):
        """Compare Watson's five live dominant emotions to the course examples."""
        for sample, expected in COURSE_CASES:
            with self.subTest(sample=sample):
                self.assertEqual(emotion_detector(sample)['dominant_emotion'], expected)


def load_tests(loader, _tests, _pattern):
    """Include the live integration check only when requested."""
    cases = [TestEmotionDetector, TestFlaskApp]
    if os.environ.get('RUN_WATSON_LIVE') == '1':
        cases.append(LiveWatsonChecks)
    return unittest.TestSuite(loader.loadTestsFromTestCase(case) for case in cases)


if __name__ == '__main__':
    unittest.main()
