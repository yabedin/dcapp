import unittest
import get_doctor_speech

class TestGetDrSpeech(unittest.TestCase):

    def test_open_json(self):
        import json 
        filepath = '../app/preprocessing/sample.json'
        sample_json = json.load(open(filepath))

        self.assertEqual(get_doctor_speech.open_json(filepath), sample_json)

        with self.assertRaises(TypeError):
            get_doctor_speech.open_json(123)

if __name__ == '__main__':
    unittest.main()
