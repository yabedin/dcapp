import unittest 
import spacy_trial


class TestSpacy(unittest.TestCase):

    def test_open_dr_words_csv(self):
        with self.assertRaises(TypeError):
            spacy_trial.convert_list_to_str(123)
        
        

if __name__ == '__main__':
    unittest.main()