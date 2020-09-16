from app import app
from flask import render_template, request, make_response, jsonify
import json
import uuid

# Import my files 
from preprocessing import convert_to_csv, get_dr_speech

@app.route("/")
def index():
    return render_template('index.html')

# API endpoint to receive JSON files from IBM Watson STT API 
@app.route('/json', methods=["POST"])
def getjson():
    '''
        Audio software sends JSON file to API ending "/json".
        Expected body: in the form of IBM Watson STT transcription 
    '''

    # Check if JSON sent
    if request.is_json:
        data = request.get_json()

        # Save incoming JSON to a CSV file. Will only save if it's in the correct format i.e. IBM Watson's format
        
        # Generate unique id for CSV filename
        unique_id = str(uuid.uuid4())[:5] 
        filename_full = unique_id + "_all_speech"
        filename_dr = unique_id + "_dr_speech"

        # Check if data is in correct format 
        if convert_to_csv.convert_to_csv(data, filename_full) == None:
            res = make_response(jsonify({"Message:":"Please send a JSON file from IBM Watson's STT API."}), 400)
            return res

        # Executes if JSON data is as expected 
        else:
            # Convert to CSV 
            convert_to_csv.convert_to_csv(data, filename_full)

            # Get dr speech 
            dr_speech_list = get_dr_speech.get_dr_speech(data, 1)
            get_dr_speech.write_csv(filename_dr, dr_speech_list)




            # Put through LR model 

            # Need to return report eventually.... 
            response = {
                "Message": "Received and CSV save"
            }

            res = make_response(jsonify(response), 200)
            return res

    # If no JSON sent
    else:
        res = make_response(jsonify({"Message:":"Please send a JSON file only."}), 400)
        return res



