try:
    from app import app
    import unittest
    import requests
    from flask import jsonify

except Exception as exception:
    print(f"Modules are missing: {exception}")

class TestFlask(unittest.TestCase):

    # Check for 200 response status code 
    def test_index(self):
        tester = app.test_client(self)
        response = tester.get("/")
        status_code = response.status_code
        self.assertEqual(status_code, 200)
    
    # Check for error message when no POST request sent
    def test_json(self):
        tester = app.test_client(self)
        response = tester.get('/json')
        self.assertEqual(response.content_type, "text/html; charset=utf-8")
        self.assertEqual(response.status_code, 405)

class TestAPI(unittest.TestCase):
    API_URL = "http://127.0.0.1:5000/json"
    sample_json =   {   "a" : "Hello",
                        "b" : "This",
                        "c" : "Is",
                        "d" : "A sample",
                        "e" : "JSON file"
                    }
    
    # Check POST request
    def test_post(self):

        sample_tuple = ('example tuple',)

        r = requests.post('http://127.0.0.1:5000/json', json=TestAPI.sample_json)
        self.assertEqual(r.status_code, 400)

if __name__ == "__main__":
    unittest.main()