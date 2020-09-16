from app import app
from flask import render_template, request, make_response, jsonify
import json
import uuid
import csv
import nltk
from nltk.corpus import stopwords
import string

# Import my files 
from preprocessing import convert_to_csv, get_dr_speech
from analysis import lr_model

@app.route("/")
def index():
    return render_template('index.html')

# API endpoint to receive JSON files from IBM Watson STT API 
@app.route('/json', methods=["POST"])
def getjson():
    '''
        Audio software sends JSON file to API ending "/json".
        Expected body: in the form of IBM Watson STT transcription WITH ADDITIONAL "dr_speaker_label": int 
        parameter specifying the doctor's speaker label transcribed. It needs to be manually labelled at this point.
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
            dr_speaker_label = data["dr_speaker_label"]
            dr_speech_list = get_dr_speech.get_dr_speech(data, dr_speaker_label)
            get_dr_speech.write_csv(filename_dr, dr_speech_list)

            # Form list from dr_speech_csv
            f = open(filename_dr + '.csv')
            csv_f = csv.reader(f)

            speech_list_for_model = []
            english_stopwords = stopwords.words('english')

            for row in csv_f:
                word = row[0]
                # Removing stop words and punctuation 
                if (word not in english_stopwords and word not in string.punctuation):
                    speech_list_for_model.append(word)
            
            # Put through LR model 
            freqs = lr_model.freqs
            theta = lr_model.theta

            y_hat = lr_model.predict_conversation(speech_list_for_model, freqs, theta)    

            if y_hat > 0.5:
                classification = 'Conversation is good'
            else: 
                classification = 'Conversation needs improving...'

            
            # Turn taking analysis 
            

            # Need to return report eventually....         
            response = {
                "Message": "Received and CSV save",
                "Output": y_hat[0][0],
                "Classification": classification
            }

            res = make_response(jsonify(response), 200)
            return res

    # If no JSON sent
    else:
        res = make_response(jsonify({"Message:":"Please send a JSON file only."}), 400)
        return res



